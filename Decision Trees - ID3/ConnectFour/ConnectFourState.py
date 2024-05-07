from __future__ import annotations
import numpy as np
from copy import (deepcopy)
from .Constants import (NROWS, NCOLS)

class Connect_Four_State:
    def __init__(self) -> None:
        # Matrix to Store the Board's Values
        self.board = np.zeros(shape=(NROWS, NCOLS), dtype=np.int8)

        # Array to keep track of the row's index for every column where a piece can be placed
        # Basically it's the Row idx for every column's height
        self.columns_height = np.full(shape=NCOLS, fill_value=(NROWS - 1), dtype=np.int8)

        # Defining the Possible Actions (Initially a piece can be placed in any column)
        self.actions = np.arange(NCOLS, dtype=np.int8)
        
        # Initializing a variable to track the current player
        self.current_player = 1
        
        # Variable to store the Winner (-1 - Game still running || 0 - Tie || 1 - Player 1 || 2 - Player 2 / AI)
        self.winner = -1
        
        # Setting a varible to store the board's move_history
        self.move_history = (self.board,)
    
    def next_player(self) -> int:
        # Returns the next turn's player
        return 3 - self.current_player

    def previous_player(self) -> int:
        # Returns the previous's player
        return self.next_player()
    
    def is_over(self) -> bool:
        # If the Winner corresponds to -1 then the game is not finished otherwise it is
        return self.winner != -1
        
    def reset(self) -> None:
        # Calls back the Constructor
        return self.__init__()
    
    def inside_board(self, x:int, y:int) -> bool:
        # Checks if a position (x,y) exists inside the board's matrix
        return (x >= 0 and x < NROWS) and (y >= 0 and y < NCOLS)
    
    def move(self, ncol:int) -> Connect_Four_State:
        # The move is not valid
        if (ncol not in self.actions):
            return self

        # Creating a new state
        new_state = deepcopy(self)
        
        # Inserting the move into the board
        nrow = new_state.columns_height[ncol]
        new_state.board[nrow, ncol] = new_state.current_player
        
        # Updating the "ncol"'s height
        new_state.columns_height[ncol] -= 1
        
        # Checking if the column is full and therefore uncapable of receiving more pieces -> Changes the action space
        if (new_state.columns_height[ncol] < 0):
            new_state.actions = np.delete(new_state.actions, np.where(new_state.actions == ncol))

        # Updates the current Winner
        new_state.update_winner(nrow, ncol)

        # Updating current player for the next state
        new_state.current_player = new_state.next_player()

        # Updating move_history
        new_state.move_history = (*new_state.move_history, new_state.board)
        
        # Returns the New State
        return new_state

    def generate_new_states(self) -> list[(int, Connect_Four_State)]:
        # List to contain all the new states
        new_states = []

        # Iterates through all possible actions and creates a new state for each one
        for ncol in self.actions:
            new_states.append((ncol, self.move(ncol)))

        # Returns all generated states
        return new_states
    
    def count_lines(self, n:int, player:int, nrow:int, ncol:int) -> bool:
        # -> Searches the Board looking for a 4-piece Combination

        # Horizontal Line
        row = nrow
        counter = 0
        for col in range(NCOLS):
            if (self.board[row, col] == player):
                counter += 1
                if (counter == n):
                    return True
            else:
                counter = 0

        # Vertical Line
        col = ncol
        counter = 0
        for row in range(NROWS):
            if (self.board[row, col] == player):
                counter += 1
                if (counter == n):
                    return True
            else:
                counter = 0

        # Descending Diagonal Line
        col = ncol
        row = nrow
        counter = 0
        while row > 0 and col > 0:
            row -= 1
            col -= 1
        while row < NROWS and col < NCOLS:
            if (self.board[row, col] == player):
                counter += 1
                if (counter == n):
                    return True
            else:
                counter = 0
            row += 1
            col += 1

        # Ascending Diagonal Line
        col = ncol
        row = nrow
        counter = 0
        while row < NROWS - 1 and col > 0:
            row += 1
            col -= 1
        while row >= 0 and col < NCOLS:
            if self.board[row, col] == player:
                counter += 1
                if counter == n:
                    return True
            else:
                counter = 0
            row -= 1
            col += 1
            
        return False
    
    def update_winner(self, nrow:int, ncol:int) -> None:
        # -> Updates the Current State's Winner
        # Checks if the Board is full already
        if (self.actions.size == 0):
            self.winner = 0
            
        # Checks for a 4-piece combination made by PLayer 1 (after he made his move)
        elif(self.current_player == 1 and self.count_lines(4, 1, nrow, ncol)):
            self.winner = 1
            
        # Checks for a 4-piece combination made by PLayer 2 (after he made his move)
        elif(self.current_player == 2 and self.count_lines(4, 2, nrow, ncol)):
            self.winner = 2

    def convert_sample_into_board(self, sample:list[list[str]]) -> Connect_Four_State.board:
        # Create a converter
        converter = {'b':0, 'x':1, 'o':2}
        vectorized_conversion = np.vectorize(converter.get)

        # Reshape the 1D Array into a 6x7 board size (Used in the Connect FOur Game)
        board =  np.array(sample).reshape(7, 6)

        # Rotate the data due to how it was stored
        corrected_board = np.rot90(board, k=1)

        # Convert the Values with the converter (in order to be used with the code developed in the Assignment #1)
        corrected_board = vectorized_conversion(corrected_board)

        # Return the final board configuration
        return corrected_board

    def convert_board_into_sample(self) -> list:
        # Create an Inverse Converter
        reverse_converter = {0:'b', 1:'x', 2:'o'}
        reverse_vectorized_conversion = np.vectorize(reverse_converter.get)

        # Rotate the data to the format it was originally in
        sample = np.rot90(self.board, k=3)

        # Convert back the values
        corrected_sample = reverse_vectorized_conversion(sample)

        # Convert the 2D Array into a 1D Array (which is a sample)
        corrected_sample = corrected_sample.flatten()

        # return the final sample (which can be used with the Decision Tree trained with the connect four dataset)
        return [corrected_sample]

    def read_state(self, file_path:str) -> Connect_Four_State:
        # Reads a game state from a text file into a new game state
        new_state = Connect_Four_State()
        
        # Creating variables to keep track of the amount of each type of pieces
        pieces_1 = 0
        pieces_2 = 0

        # Reading the board from the text file
        with open(file_path, "r") as f:
            lines = [line.rstrip() for line in f]

        # Updates the Board Matrix
        for i in range(len(lines)):
            for j in range(len(lines[i])):
                if (lines[i][j] == '-'):
                    new_state.board[i, j] = 0
                elif (lines[i][j] == 'X'):
                    new_state.board[i, j] = 1
                    pieces_1 += 1
                elif (lines[i][j] == 'O'):
                    new_state.board[i, j] = 2
                    pieces_2 += 1

        # Updating the Column's Heights
        for ncol in range(len(new_state.actions)):
            for nrow in range(NROWS - 1, 0, -1):
                if (new_state.board[nrow, ncol] != 0):
                    if (new_state.inside_board(ncol, nrow - 1)):
                        new_state.columns_height[ncol] = nrow - 1
                    else:
                        new_state.actions = np.delete(new_state.actions, np.where(new_state.actions == ncol))
        
        # Updates next player
        if (pieces_1 > pieces_2):
            new_state.current_player = 2
        else:
            new_state.current_player = 1

        # Updating move_history
        new_state.move_history = (new_state.board, )
        
        return new_state
    
    """ AUXILIAR METHODS """

    def __str__(self) -> str:
        # -> Converts the board into the style used in the Assignment 1 Paper
        DECODER = {0:'-', 1:'X', 2:'O'}
        line = ["-" for i in range(2*NCOLS -1)]
        line.insert(0, '#')
        line.insert(1, ' ')
        line.insert(len(line), ' ')
        line.insert(len(line), '#')
        formated_line = "".join(line)
        new_board = formated_line + '\n'
        for x in range (NROWS):
            for y in range (NCOLS):
                if (y == 0):
                    new_board += "| " + DECODER[self.board[x, y]]
                elif (y == NCOLS -1):
                    new_board += " " + DECODER[self.board[x, y]] + " |"
                else:
                    new_board += " " + DECODER[self.board[x, y]]
            new_board += '\n'
        new_board += formated_line
        return new_board
        
    def __hash__(self) -> hash:
        return hash(str(self.board))

    def __eq__(self, other:object) -> bool:
        if (not isinstance(other, Connect_Four_State)):
            raise Exception(f"Sorry, other object is not an instance of {self.__class__.__name__}")
        return hash(self) == hash(other)
    
if __name__ == "__main__":
    game = Connect_Four_State()
    print(game, "\n")
    game = game.move(2)
    game = game.move(2)
    game = game.move(2)
    game = game.move(2)
    game = game.move(2)
    game = game.move(2)
    game = game.move(2)
    print(game, "\n")