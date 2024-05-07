# This File contains the Suggested Heuristic from the Assignment #1

from .Constants import (NROWS, NCOLS)
from .ConnectFourState import (Connect_Four_State)
from .TreeNode import (TreeNode)

def check_line_configuration(player:int, values:list[int]) -> tuple[int]:
    # Counts the amount of pieces of each type
    
    # List with the amount of pieces for each type
    pieces = [0, # Empty Spaces - Index 0
              0, # Player 1 Pieces - Index 1
              0] # Player 2 Pieces - Index 2
    
    # Doing a Linear Search throughout the given sequence
    for piece_type in values:
        pieces[piece_type] += 1

    # Considering the list configuration we return the amount of pieces from each type
    if (player == 1): # Previous Player => Player 1
        return (pieces[1], pieces[2], pieces[0])
    else: # Previous Player => Player 2
        return (pieces[2], pieces[1], pieces[0])

def calculate_line_score(player_pieces:int, enemy_pieces:int, empty_spaces:int, state:Connect_Four_State, extra=False) -> int:
    # -> Calculates the score to return based on the line configuration
    # Defining a Score Decoder for the amount of Empty Spaces
    SCORE_DECODER = [512,  # Idx - 0 Empty Spaces - There are 4 player's pieces
                     50,   # Idx - 1 Empty Space  - There are 3 player's pieces
                     10,   # Idx - 2 Empty Spaces - There are 2 player's pieces
                     1,    # Idx - 3 Empty Spaces - There is 1 player's pieces
                     0]    # Idx - 4 Empty Spaces - There are only empty spaces

    # Initializing the Score that is going to be returned
    score = 0

    # There are player pieces
    if (player_pieces > 0):
        # We have both player's pieces
        if (enemy_pieces > 0):
            score = 0
        # There are no enemy pieces
        else:
            score = SCORE_DECODER[empty_spaces]
    else:
        score = - SCORE_DECODER[empty_spaces]

    # Returning final score evaluation for the 4-piece sequence
    return score

def calculate_score(state:Connect_Four_State) -> int:
    # -> Calculates current State Evaluation [Based on the Assignment's Suggestion]
    # Initializes the number of lines found
    total_score = 0

    # In order to evaluate a node's heuristic we have to keep in mind the previous player 
    # because this node's score is the one that is going to influence the parent's choice upon the children
    player = state.previous_player()
    
    # Adding the Handicap
    if (player == 1):
        total_score -= 16
    else:
        total_score += 16
    
    # Loops through the board
    for row in range(NROWS):
        for col in range(NCOLS):
            # Checks a Horizontal Line
            if col < NCOLS - 3:
                (player_pieces, enemy_player_pieces, empty_spaces) = check_line_configuration(player, [state.board[row, col + i] for i in range(4)])
                total_score += calculate_line_score(player_pieces, enemy_player_pieces, empty_spaces, state)
                
            # Checks a Vertical Line
            if row < NROWS - 3:
                (player_pieces, enemy_player_pieces, empty_spaces) = check_line_configuration(player, [state.board[row + i, col] for i in range(4)])
                total_score += calculate_line_score(player_pieces, enemy_player_pieces, empty_spaces, state)
            
            # Checks a Descending Diagonal Line
            if row < NROWS - 3 and col < NCOLS - 3:
                (player_pieces, enemy_player_pieces, empty_spaces) = check_line_configuration(player, [state.board[row + i, col + i] for i in range(4)])
                total_score += calculate_line_score(player_pieces, enemy_player_pieces, empty_spaces, state)
    
            # Checks a Ascending Diagonal Line
            if col < NCOLS - 3 and row > 3:
                (player_pieces, enemy_player_pieces, empty_spaces) = check_line_configuration(player, [state.board[row - i, col + i] for i in range(4)])
                total_score += calculate_line_score(player_pieces, enemy_player_pieces, empty_spaces, state)
            
    return total_score

def heuristic_suggested(state:TreeNode) -> int:
    # Suggested Heuristic in the Assignment I Paper
    return calculate_score(state)