# Defining some Constants

# Defining Board's Size
NROWS = 6
NCOLS = 7

# Some Parameters for the Graphical User Interface
# Screen Parameters
SQSIZE = 80
X_OFFSET = 60
Y_OFFSET = 100
BORDER_THICKNESS = 10
WIDTH = NCOLS*SQSIZE + 2*X_OFFSET
HEIGHT = NROWS*SQSIZE + 2*Y_OFFSET

# Circle - Screen Parameters
CIRCLE_OFFSET = 10
CIRCLE_POS = (X_OFFSET + SQSIZE//2, Y_OFFSET + SQSIZE//2)
CIRCLE_RADIUS = (SQSIZE//2) - CIRCLE_OFFSET

# RGB Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

LIGHT_BLUE = (135, 206, 250)
BLUE = (102, 178, 255)
DARK_BLUE = (0, 76, 153)

RED = (189, 22, 44)
DARK_RED = (151, 18, 35)

GREEN = (0, 204, 102)
DARK_GREEN = (0, 153, 76)

# Defining a Array with the Piece's Colors [Tuple: (BORDER, INNER CIRCLE)]
PIECES_COLORS = [(DARK_BLUE, BLUE),        # Empty Pieces
                (DARK_RED, RED),          # Player 1
                (DARK_GREEN, GREEN)]      # Player 2