from __future__ import annotations
import random as rd
from .ConnectFourState import (Connect_Four_State)
from .TreeNode import (TreeNode)
from .Heuristics import (heuristic_suggested)
from .Algorithms import (A_Star_Search, MiniMax, MCTS)
from ..ID3.ID3 import (DecisionTree)
from IPython.display import (clear_output)

class Connect_Four_Terminal_APP:
    def __init__(self, ConnectFour_DT:DecisionTree) -> None:
        self.current_node = TreeNode(state=Connect_Four_State())
        self.menu = "Main_Menu"

        # Storing a Trained Decision tree on the Connect Four Dataset
        self.dt = ConnectFour_DT

    """ Player & Algorithms """
    def player(self, show=True) -> TreeNode:
        # Printing current board configuration
        print(f"\n  CURRENT BOARD \n{self.current_node.state}")
        
        # Requesting a column to play at
        ncol = int(input(f"\n| Player {self.current_node.state.current_player} | Choose a Column to Play: "))
        
        # Creating a new Node by making a move into the "ncol" column
        new_node = self.current_node.generate_new_node(ncol)
    
        return new_node

    def random(self, show=True) -> TreeNode:
        # Randomizing a col to play at
        ncol_idx = rd.randrange(0, len(self.current_node.state.actions))
        ncol = self.current_node.state.actions[ncol_idx]
        
        if (show):
            # Printing current board configuration
            print(f"\n  CURRENT BOARD \n{self.current_node.state}")
            print(f"\n| Random AI | Played in the {ncol}th column ")
    
        # Creating a new Node by making a move into the "ncol" column
        new_node = self.current_node.generate_new_node(ncol)

        return new_node

    def A_Star_action(self, heuristic:function) -> TreeNode:
        # Getting the Final Node after using the A* Search
        final_node = A_Star_Search(self.current_node, heuristic)
        
        # return next node
        return final_node
    
    def A_Star(self, heuristic=heuristic_suggested, show=True) -> TreeNode:
        if (show):
            # Printing current board configuration
            print(f"\n  CURRENT BOARD \n{self.current_node.state}")
            print(f"\n| A* Search | Played ")
        
        # Generate the Next_Node
        new_node = self.A_Star_action(heuristic)
    
        # Returns the next node
        return new_node

    def minimax(self, heuristic=heuristic_suggested, depth_search=5, show=True) -> TreeNode:
        # Executing a MiniMax move with both heuristics and depth search given
        return MiniMax(self.current_node, heuristic, depth_search)

    def mcts(self, heuristic=heuristic_suggested, show=True) -> TreeNode:
        # Executing the Monte Carlo Tree Search Algorithm
        return MCTS(self.current_node, heuristic)
    
    """ GAME LOOP """
    def run_game(self, player1:int, player2:int, heuristic_1=None, heuristic_2=None, show_output=True) -> int:
        clear_output() # Clearing the Cell's Output
        
        self.current_node = TreeNode(state=Connect_Four_State()) # Reset the Board
        
        while not self.current_node.state.is_over():
            # Player 1
            if self.current_node.state.current_player == 1:
                if heuristic_1 is None:
                    # print("Unexplored actions before removal:", self.current_node.unexplored_actions)
                    new_node = player1(show=show_output)                    
                    self.current_node = new_node
                    # print("Unexplored actions after removal:", self.current_node.unexplored_actions)
                else:
                    new_node = player1(heuristic=heuristic_1, show=show_output)
                    self.current_node = new_node

            # Player 2
            else:
                if heuristic_2 is None:
                    new_node = player2(show=show_output)
                    self.current_node = new_node
                else:
                    new_node = player2(heuristic=heuristic_2, show=show_output)
                    self.current_node = new_node

        if(show_output):
            # Printing Final Board Configuration
            print(f"\n   FINAL BOARD\n{self.current_node.state}")
            
            if self.current_node.state.winner == 0: # Checking if it was a Tie
                print("\n-> Tie")
            elif self.current_node.state.winner == 1: # Approach 1 Won
                print(f"\n-> {player1.__name__} {self.current_node.state.winner} Wins!")
            else: # Approach 2 Won
                print(f"\n-> {player2.__name__} {self.current_node.state.winner} Wins!")
    
            if (self.to_continue()):
                self.menu = "Main_Menu"
            else:
                self.menu = "EXIT"
        
        return self.current_node.state.winner

    def run_multiple_games(self, n_games:int, player1:int, player2:int, heuristic_1=None, heuristic_2=None, show_output=True) -> list[int]:
        # Creating a list to store the results
        results = [0, 0, 0]
        for _ in range(n_games):
            # Running a certain game n times for evaluation purposes
            winner = self.run_game(player1, player2, heuristic_1, heuristic_2, show_output)
            # Updating the Results
            results[winner] += 1
        # Returning the results
        return results

    def show_multiple_games_results(self, player1_name:str, player2_name:str, results:list[int], n_games:int) -> None:
        print("#-------------------------#")
        print("|   # Results Analysis    |")
        print("#-------------------------#\n")
        print(f"-> {player1_name} \tWON {results[1]} MATCHES")
        print(f"-> {player2_name} \tWON {results[2]} MATCHES")
        print(f"-> THERE WERE \t{results[0]} TIES\n")
        
        dashed_line_length = len(" TOTAL MATCHES:  ") + len(str(n_games))
        line = ['-' for _ in range(dashed_line_length)]
        line.insert(0, '#')
        line.insert(len(line), '#')
        formated_line = "".join(line)
        
        print(formated_line)
        print(f"| TOTAL MATCHES: {n_games} |")
        print(formated_line)
        
    def to_continue(self) -> bool:
        choice = input("\nWould you like to Continue? [y/n] : ")
        while (choice.lower() != "y" and choice.lower() != "n"):
            choice = input("\nWould you like to Continue? [y/n] : ")
        if (choice.lower() == "y"):
            return True
        return False

    def menus_base_function(self, print_function:function, lower_value:int, higher_value:int, multiple_values=False, back_item=5) -> int:
        clear_output()
        print_function()
        if (multiple_values):
            options = list(map(int, input("  OPTIONS: ").split()))
            if (len(options) == 1 and options[0] == back_item):
                self.menu = "Main_Menu"
            elif (len(options) == 1 and options[0] == 0):
                self.menu = "EXIT"
            elif (len(options) != 2 or (options[0] < lower_value or options[0] > higher_value) or (options[1] < lower_value or options[1] > higher_value)):
                return self.menus_base_function(print_function, lower_value, higher_value, multiple_values, back_item)
            return options
        else:
            option =  int(input("  OPTION: "))
            if (option < lower_value or option > higher_value):
                return self.menus_base_function(print_function, lower_value, higher_value)
            return option
    
    def print_main_menu(self) -> None:
        print("# --------------------------------- #")
        print("|             MAIN MENU             |")
        print("# --------------------------------- #")
        print("| 1 - Player vs Algorithms          |")
        print("# --------------------------------- #")
        print("| 0 - EXIT                          |")
        print("# --------------------------------- #")
        
    def main_menu(self) -> int:
        return self.menus_base_function(print_function=self.print_main_menu, lower_value=0, higher_value=1)

    def print_player_vs_algorithms(self) -> None:
        print("# --------------------------------- #")
        print("|       CHOOSE YOUR OPPONENT        |")
        print("# --------------------------------- #")
        print("| 1 - Random Choice                 |")
        print("| 2 - A* Search                     |")
        print("| 3 - MiniMax                       |")
        print("| 4 - Monte Carlo Tree Search       |")
        print("|                                   |")
        print("| 5 - Back                          |")
        print("# --------------------------------- #")
        print("| 0 - EXIT                          |")
        print("# --------------------------------- #")
    
    def player_vs_algorithms(self) -> int:
        return self.menus_base_function(print_function=self.print_player_vs_algorithms, lower_value=0, higher_value=5)
    
    def id3_heuristic_V1(self, state:Connect_Four_State) -> int:
        # Converts the board into a sample to be fed to the algorithm
        sample = state.convert_board_into_sample()

        # Predicting the outcome of the board configuration
        y_pred = self.dt.predict(sample)[0]

        # We need a conversion since the Label Encoder labeled the target classes as {Draw:0, Lose:1, Win:2}
        convert = {0:1, 1:0, 2:2}

        return convert[y_pred]
    
    def id3_heuristic_V2(self, state:Connect_Four_State) -> int:
        # Converts the board into a sample to be fed to the algorithm
        sample = state.convert_board_into_sample()

        # Calculates the probability of each possible end game
        [[prob_draw, prob_lose, prob_win]] = self.dt.predict_proba(sample)

        # Add some coefficents to try to improve the results
        total_pred = -1000*prob_lose + 5*prob_draw + 10*prob_win

        return total_pred

    def run(self) -> None:
        self.menu = "Main_Menu"
        
        while self.menu != "EXIT":
            if (self.menu == "Main_Menu"):
                        
                option = self.main_menu()
                
                if (option == 1): self.menu = "Player_vs_Algorithms"
                else: self.menu = "EXIT"
            
            elif (self.menu == "Player_vs_Algorithms"):

                option = self.player_vs_algorithms()
                
                if (option == 1): # Random Choice Game
                    self.run_game(player1=self.random, player2=self.player, heuristic_1=None, heuristic_2=None, show_output=True)
                
                elif (option == 2): # A* Search
                    self.run_game(player1=self.A_Star, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None, show_output=True)
                
                elif (option == 3): # MiniMax
                    self.run_game(player1=self.minimax, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None, show_output=True)
                
                elif (option == 4): # Monte Carlo Tree Search
                    self.run_game(player1=self.mcts, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None, show_output=True)
                
                elif (option == 5): # BACK
                    self.menu = "Main_Menu"
               
                else: # EXIT
                    self.menu = "EXIT"

            else:
                self.menu = "EXIT"