import random as rd
from IPython.display import (clear_output)
from .ConnectFourState import (Connect_Four_State)
from .TreeNode import (TreeNode)

class Connect_Four_Terminal_APP:
    def __init__(self):
        self.current_node = TreeNode(state=Connect_Four_State())
        self.menu = "Main_Menu"

    """ Player & Algorithms """
    def player(self, show=True):
        # Printing current board configuration
        print(f"\n  CURRENT BOARD \n{self.current_node.state}")
        
        # Requesting a column to play at
        ncol = int(input(f"\n| Player {self.current_node.state.current_player} | Choose a Column to Play: "))
        
        # Creating a new Node by making a move into the "ncol" column
        new_node = self.current_node.generate_new_node(ncol)
    
        return new_node

    def random(self, show=True):
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
    
    def id3_move(self):
        print("TO BE IMPLEMENTED")

    """ GAME LOOP - Meant to be used inside a Notebook - Be Aware of the clear_output funtion"""
    def run_game(self, player1, player2, heuristic_1=None, heuristic_2=None, show_output=True):
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
        
    def to_continue(self):
        choice = input("\nWould you like to Continue? [y/n] : ")
        while (choice.lower() != "y" and choice.lower() != "n"):
            choice = input("\nWould you like to Continue? [y/n] : ")
        if (choice.lower() == "y"):
            return True
        return False

    def menus_base_function(self, print_function, lower_value, higher_value, multiple_values=False, back_item=3):
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
    
    def print_main_menu(self):
        print("# --------------------------------- #")
        print("|             MAIN MENU             |")
        print("# --------------------------------- #")
        print("| 1 - Player vs Algorithms          |")
        print("# --------------------------------- #")
        print("| 0 - EXIT                          |")
        print("# --------------------------------- #")
        
    def main_menu(self):
        return self.menus_base_function(print_function=self.print_main_menu, lower_value=0, higher_value=1)

    def print_player_vs_algorithms(self):
        print("# --------------------------------- #")
        print("|       CHOOSE YOUR OPPONENT        |")
        print("# --------------------------------- #")
        print("| 1 - Random Choice                 |")
        print("| 2 - ID3 Algorithm                 |")
        print("|                                   |")
        print("| 3 - Back                          |")
        print("# --------------------------------- #")
        print("| 0 - EXIT                          |")
        print("# --------------------------------- #")
    
    def player_vs_algorithms(self):
        return self.menus_base_function(print_function=self.print_player_vs_algorithms, lower_value=0, higher_value=5)
    
    def execute(self):
        self.menu = "Main_Menu"
        
        while self.menu != "EXIT":
            if (self.menu == "Main_Menu"):
                        
                option = self.main_menu()
                
                if (option == 1): self.menu = "Player_vs_Algorithms"
                else: self.menu = "EXIT"
            
            elif (self.menu == "Player_vs_Algorithms"):

                option = self.player_vs_algorithms()
                
                if (option == 1): # Random Choice Game
                    self.run_game(player1=self.player, player2=self.random, heuristic_1=None, heuristic_2=None, show_output=True)
                
                elif (option == 2): # ID3 Algorithm
                    print("TO BE IMPLEMENTED")
                
                elif (option == 3): # BACK
                    self.menu = "Main_Menu"
               
                else: # EXIT
                    self.menu = "EXIT"

            else:
                self.menu = "EXIT"