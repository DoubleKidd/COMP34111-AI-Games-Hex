from enum import Enum


class Colour(Enum):
    """This enum describes the sides in a game of Hex."""

    # RED is vertical, BLUE is horizontal
    RED = 0
    BLUE = 1

    def get_char(colour):
        """Returns the name of the colour as an uppercase character."""

        if colour == Colour.RED:
            return "\033[31m█\033[0m"
        elif colour == Colour.BLUE:
            return "\033[34m█\033[0m"
        else:
            return "·"

    def from_char(c):
        """Returns a colour from its char representations."""

        if c == "\033[31m█\033[0m":
            return Colour.RED
        elif c == "\033[34m█\033[0m":
            return Colour.BLUE
        else:
            return None

    def opposite(colour):
        """Returns the opposite colour."""

        if colour == Colour.RED:
            return Colour.BLUE
        elif colour == Colour.BLUE:
            return Colour.RED
        else:
            return None


if __name__ == "__main__":
    for colour in Colour:
        print(colour)
