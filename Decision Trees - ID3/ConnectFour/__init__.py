# This Python Package contains code that was previously implemented on the Project #1 Assignment

import os
import sys

# Function to determine the base path of the package (handles frozen and normal environments)
def get_base_dir():
    if getattr(sys, 'frozen', False):
        return sys._MEIPASS
    else:
        # Return the directory where this file is located
        return os.path.dirname(os.path.abspath(__file__))

# Use the base path to define a function that gives asset paths
def get_asset_path(folder, asset_filename):
    base_dir = get_base_dir()
    return os.path.join(base_dir, folder, asset_filename)

from .ConnectFourState import (Connect_Four_State)
from .TerminalInterface import (Connect_Four_Terminal_APP)
from .GraphicalUserInterface import (Connect_Four_GUI_APP)