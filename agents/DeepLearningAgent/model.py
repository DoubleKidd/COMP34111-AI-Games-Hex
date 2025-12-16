import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard ResNet block.
    It passes the input through two convolutions and adds the original input back
    to the result (skip connection). This prevents the signal from degrading.
    """
    def __init__(self, num_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # The "Skip Connection"
        out = F.relu(out)
        return out

class AlphaZeroHexNet(nn.Module):
    def __init__(self, board_size=11, num_res_blocks=4, num_channels=64):
        super().__init__()
        self.board_size = board_size
        self.num_channels = num_channels
        
        # --- Initial Block ---
        # Takes 2 input channels (My stones, Opp stones) -> Outputs 64 feature maps
        # This is the bit that checks kernal_size x kernal_size areas of the board to produce relationship maps.
        # This converts the 0/1 values into a richer feature representation including neighbourhood info.
        self.conv_input = nn.Conv2d(2, num_channels, kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)

        # --- Residual Tower ---
        # A stack of residual blocks to "reason" about the board
        # Effectively a deep feature extractor that builds up complex features from simple ones.
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])

        # --- Policy Head (The "Move" predictor) ---
        # Reduces channels to 2, then flattens to a vector of size 11*11
        # Uses the same kernal style as in the inputs to produce a move probability distribution,
        # where each position on the board has a likelihood of being chosen.
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        # --- Value Head (The "Win" predictor) ---
        # Reduces channels to 1, flattens, then reduces to a single number
        # Uses the same kernal style as in the inputs to produce a value estimate,
        # which predicts how likely the current player is to win from this position.
        # -1.0 = Lose, 0 = Draw, 1.0 = Win
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch_size, 2, 11, 11)
        
        # 1. Initial Conv
        x = F.relu(self.bn_input(self.conv_input(x)))

        # 2. Residual Tower
        for block in self.res_blocks:
            x = block(x)

        # 3. Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)  # Flatten
        p = self.policy_fc(p)
        
        # LogSoftmax is better for numerical stability during training.
        # When playing, we will exponentiate this to get real probabilities.
        p = F.log_softmax(p, dim=1) 

        # 4. Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)  # Flatten
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)) # Output between -1 (Lose) and 1 (Win)

        return p, v

# --- SANITY CHECK ---
if __name__ == "__main__":
    # Create a random fake board batch (1 game, 2 channels, 11x11)
    dummy_input = torch.randn(1, 2, 11, 11)
    
    model = AlphaZeroHexNet()
    policy, value = model(dummy_input)
    
    print("Policy Shape:", policy.shape) # Should be [1, 121]
    print("Value Shape:", value.shape)   # Should be [1, 1]
    
    # Check if value is between -1 and 1
    print(f"Value Output: {value.item():.4f}")
    assert -1 <= value.item() <= 1
    print("Test Passed!")