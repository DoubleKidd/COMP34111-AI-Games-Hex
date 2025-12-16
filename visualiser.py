import tkinter as tk
import math

# Board size
ROWS = 11
COLS = 11
HEX_SIZE = 30

# Neighbor directions
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1)]

class HexGUI:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=900, height=900, bg='white')
        self.canvas.pack()

        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.hex_ids = {}

        self.draw_board()
        self.canvas.bind('<Button-1>', self.left_click)
        self.canvas.bind('<Button-3>', self.right_click)

    def hex_corners(self, x, y):
        corners = []
        for i in range(6):
            angle = math.radians(60 * i + 30)
            cx = x + HEX_SIZE * math.cos(angle)
            cy = y + HEX_SIZE * math.sin(angle)
            corners.append((cx, cy))
        return corners

    def draw_hex(self, row, col, color='white'):
        x = 100 + col * HEX_SIZE * 1.5
        y = 100 + row * HEX_SIZE * math.sqrt(3) - col * (HEX_SIZE * math.sqrt(3) / 2)

        corners = self.hex_corners(x, y)
        hex_id = self.canvas.create_polygon(corners, fill=color, outline='black', width=2)
        self.hex_ids[hex_id] = (row, col)
        return hex_id

    def draw_board(self):
        # Draw main grid
        for r in range(ROWS):
            for c in range(COLS):
                self.draw_hex(r, c)

        # Draw top/bottom red walls
        for c in range(COLS):
            self.draw_hex(-1, c, color='red')
            self.draw_hex(ROWS, c, color='red')

        # Draw left/right blue walls
        for r in range(ROWS):
            self.draw_hex(r, -1, color='blue')
            self.draw_hex(r, COLS, color='blue')

    def left_click(self, event):
        self.handle_click(event, 'blue')

    def right_click(self, event):
        self.handle_click(event, 'red')

    def handle_click(self, event, color):
        clicked = self.canvas.find_closest(event.x, event.y)[0]
        if clicked in self.hex_ids:
            r, c = self.hex_ids[clicked]
            if 0 <= r < ROWS and 0 <= c < COLS:
                self.canvas.itemconfig(clicked, fill=color)
                self.board[r][c] = color


root = tk.Tk()
root.title("Hex GUI")
app = HexGUI(root)
root.mainloop()
