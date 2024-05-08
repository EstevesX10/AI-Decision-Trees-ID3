import numpy as np
import random as rd
import pygame
from typing import (Callable)
from . import (get_asset_path)
from .Constants import (NROWS, NCOLS,
                       SQSIZE, X_OFFSET, Y_OFFSET, BORDER_THICKNESS, WIDTH, HEIGHT,
                       CIRCLE_OFFSET, CIRCLE_POS, CIRCLE_RADIUS,
                       BLACK, WHITE, LIGHT_BLUE, BLUE, DARK_BLUE, RED, DARK_RED, GREEN, DARK_GREEN, PIECES_COLORS)
from .ConnectFourState import (Connect_Four_State)
from .TreeNode import (TreeNode)
from .Heuristics import (heuristic_suggested)
from .Algorithms import (A_Star_Search, MiniMax, MCTS)
from ID3.ID3 import (DecisionTree)

# NOTE:
# -> Both Image and Button Classes are almost the same. They only differ upon the button's functionality since a Image is only used to display a sprite while the button 
# can be pressed and trigger some other functions

class Image:
    def __init__(self, image, x, y, scale):
        self.Height = image.get_height() # Defining Image's Height
        self.Width = image.get_width() # Defining Image's Width
        self.scale = scale # Defining a Scale to which the sprite will be resized
        self.image = pygame.transform.scale(image, (int(self.Width*self.scale), int(self.Height*self.scale))) # Resizing the Sprite
        self.rect = self.image.get_rect() # Creating a Rectangle for the Image's Sprite
        self.rect.topleft = (x,y) # Defining the Position where the image must be placed at

    def Display(self, screen):
        screen.blit(self.image, (self.rect.x, self.rect.y)) # Displaying the Image into the given Screen

class Button:
    def __init__(self, image, x, y, scale):
        self.Height = image.get_height() # Defining Button's Height
        self.Width = image.get_width() # Defining Button's Width
        self.scale = scale # Defining a Scale to which the sprite will be resized
        self.image = pygame.transform.scale(image, (int(self.Width*self.scale), int(self.Height*self.scale))) # Resizing the Sprite
        self.rect = self.image.get_rect() # Creating a Rectangle for the Button's Sprite
        self.rect.topleft = (x,y) # Defining the Position where the button must be placed at
        self.clicked = False # Flag that determines if the button has been clicked

        self.FirstContact = 0 # State of the Mouse when he fisrtly approached a button's region
        self.NumContacts = 0 # Total contacts the mouse has made with the button


    def Action(self, Tela):
        Action = False # Flag to determine if the button has been activated
        Mouse_Pos = pygame.mouse.get_pos() # Gets Mouse Position

        if self.rect.collidepoint(Mouse_Pos): # Checks if the mouse position collides with the Button sprite

            if pygame.mouse.get_pressed()[0] == 0: # If the Mouse is not clicking inside a button's region then we can reset the Number of Contacts
                self.clicked = False
                self.NumContacts = 0

            if (self.NumContacts == 0): # Checks if it's the first contact between the mouse and the button [and Stores the Mouse's state]
                self.FirstContact = (pygame.mouse.get_pressed()[0])

            if pygame.mouse.get_pressed()[0] == 1 and self.clicked == False:
                if (self.FirstContact == 0): # The mouse did not reach the Sprite's Area while being pressed on
                    self.clicked = True
                    Action = True
                
            self.NumContacts += 1 # If the Mouse is above the Button then we increment the Number of Contacts

        else: # Resets total Contacts
            self.NumContacts = 0

        Tela.blit(self.image, (self.rect.x, self.rect.y)) # Inserting the Sprite into the Screen
        return Action # Returning if the button was activated
    
class Connect_Four_GUI_APP:
    def __init__(self, ConnectFour_DT:DecisionTree) -> None:
        # Initializing the current_node with a initial state
        self.current_node = TreeNode(state=Connect_Four_State())

        # Flag to keep track of the clicks
        self.clicked = True

        # Declaring a variable to keep track of the current menu
        self.menu = "Main_Menu"

        # Storing a Trained Decision tree on the Connect Four Dataset
        self.dt = ConnectFour_DT

    """ PLAYER ACTIONS & ALGORITHMS"""
    def player(self) -> TreeNode:
        # In case we don't do any move the node stays the same
        new_node = self.current_node
        
        # Checking if we pressed the mouse 1 button and therefore changed the self.clicked flag
        if not self.clicked and pygame.mouse.get_pressed()[0] == 1:
            
            # Getting Mouse Position
            (y, x) = pygame.mouse.get_pos()
    
            # Modifying the Mouse Coordinates to match with the Screen Stylling
            x = (x - Y_OFFSET) // SQSIZE
            y = (y - X_OFFSET) // SQSIZE
    
            # Checking if the coordenates exist in the board. If so, add a piece to the column that the mouse was pressed on
            if (self.current_node.state.inside_board(x, y)):
                new_node = self.current_node.generate_new_node(y)
    
            # Updating the "clicked" flag
            self.clicked = True
    
        # Checking if we released the mouse 1 button and therefore changed the self.clicked flag
        if self.clicked and pygame.mouse.get_pressed()[0] == 0:
            
            # Updating the "clicked" flag
            self.clicked = False
    
        return new_node

    def random(self) -> TreeNode:
        # Randomizing a col to play at & printing which one it was
        ncol_idx = rd.randrange(0, len(self.current_node.state.actions))
        ncol = self.current_node.state.actions[ncol_idx]
    
        # Creating a new Node by making a move into the "ncol" column
        new_node = self.current_node.generate_new_node(ncol)
    
        return new_node

    def A_Star_action(self, heuristic:Callable[[TreeNode], int]) -> TreeNode:
        # Getting the Final Node after using the A* Search
        final_node = A_Star_Search(self.current_node, heuristic)
        
        # return next node
        return final_node
    
    def A_Star(self, heuristic=heuristic_suggested) -> TreeNode:
        # Generate the Next_Node
        new_node = self.A_Star_action(heuristic)
    
        # Returns the next node
        return new_node

    def minimax(self, heuristic:Callable[[TreeNode], int], depth_search=5) -> TreeNode:
        # Executing a MiniMax move with both heuristics and depth search given
        return MiniMax(self.current_node, heuristic, depth_search)

    def mcts(self, heuristic:Callable[[TreeNode], int]) -> TreeNode:
        # Executing the Monte Carlo Tree Search Algorithm
        return MCTS(self.current_node, heuristic)

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

    def id3(self) -> TreeNode:
        best_col = -1
        best_proba_win = best_proba_draw = -1
        best_proba_lose = 1000

        for (n_col, new_state) in self.current_node.state.generate_new_states():
            sample = new_state.convert_board_into_sample()
            [[prob_draw, prob_lose, prob_win]] = self.dt.predict_proba(sample)

            if (prob_win > best_proba_win):
                best_proba_win = prob_win
                best_proba_draw = prob_draw
                best_proba_lose = prob_lose
                best_col = n_col
            elif (prob_win == best_proba_win):
                if (prob_draw > best_proba_draw):
                    best_proba_draw = prob_draw
                    best_proba_lose = prob_lose
                    best_col = n_col
                elif (prob_draw == best_proba_draw):
                    if (prob_lose < best_proba_lose):
                        best_proba_lose = prob_lose
                        best_col = n_col

        # Creating a new Node by making a move into the "ncol" column
        new_node = self.current_node.generate_new_node(best_col)

        return new_node

    """ GUI METHODS """
    def write(self, font:str, text:str, size:tuple, color:str, bg_color:str, bold:bool, pos:tuple, screen:pygame.Surface) -> None:
        # Writes Text into the Screen
        letra = pygame.font.SysFont(font, size, bold)
        frase = letra.render(text, 1, color, bg_color)
        screen.blit(frase, pos)

    def write_winner(self, screen:pygame.Surface, winner_name:str) -> None:
        if (self.current_node.state.winner != 0):
            winner_text = (" " + winner_name + " " + str(self.current_node.state.winner) + " Wins! ")
        else:
            winner_text = " Tie! "
        font_size = 45
        winner_text_length = len(winner_text)
        (x, y) = ((WIDTH  - winner_text_length)//2, (Y_OFFSET - font_size - BORDER_THICKNESS) // 2)
        self.write(font='Arial', text=winner_text, size=font_size, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(0.65*x, 25), screen=screen)
    
    def draw_board(self, screen:pygame.Surface) -> None:
        # Draws Board's Shadow
        board_rect_shadow = pygame.Rect((X_OFFSET - BORDER_THICKNESS, Y_OFFSET - BORDER_THICKNESS),
                                        (SQSIZE*NCOLS + 2*BORDER_THICKNESS, SQSIZE*NROWS + 2*BORDER_THICKNESS))
        pygame.draw.rect(screen, DARK_BLUE, board_rect_shadow)

        # Draws Main Board
        board_rect = pygame.Rect((X_OFFSET, Y_OFFSET), (SQSIZE*NCOLS, SQSIZE*NROWS))
        pygame.draw.rect(screen, BLUE, board_rect)

        # Drawing Circles in the Board
        for row in range(NROWS):
            for col in range(NCOLS):
                # Getting the Colors from the Auxiliar List
                (Border_Color, Circle_Color) = PIECES_COLORS[self.current_node.state.board[row,col]]

                # Drawing the Board's border around the pieces
                pygame.draw.circle(screen, DARK_BLUE, (X_OFFSET + SQSIZE//2 + (col*SQSIZE),
                                                       Y_OFFSET + SQSIZE//2 + (row*SQSIZE)), int(1.15*CIRCLE_RADIUS))
                
                # Drawing the Circle's Border
                pygame.draw.circle(screen, Border_Color, (X_OFFSET + SQSIZE//2 + (col*SQSIZE),
                                                          Y_OFFSET + SQSIZE//2 + (row*SQSIZE)), CIRCLE_RADIUS)

                # Drawing the Main Circle
                pygame.draw.circle(screen, Circle_Color, (X_OFFSET + SQSIZE//2 + (col*SQSIZE),
                                                          Y_OFFSET + SQSIZE//2 + (row*SQSIZE)), int(0.9*CIRCLE_RADIUS))
    
    def draw(self, screen:pygame.Surface) -> None:
        # Filling the Background with Blue
        screen.fill(LIGHT_BLUE)

        # Drawing the Current Board Elements
        self.draw_board(screen)
    
    def run_game(self, screen:pygame.Surface, player1:int, player2:int, heuristic_1=None, heuristic_2=None) -> int:
        # Reseting the game
        self.current_node = TreeNode(state=Connect_Four_State())

        # Creating Buttons
        BACK_PATH = get_asset_path('Assets', 'Back.png')
        BACK_IMG = pygame.image.load(BACK_PATH).convert_alpha()
        Back_Btn = Button(BACK_IMG, 20, 20, 0.1)
        
        # Create a Flag to keep track of current state of the Application / GUI
        game_run = True
        
        # Main Loop
        while game_run:
            
            # Draws the Game Elements into the Screen
            self.draw(screen)

            if Back_Btn.Action(screen):
                game_run = False
            
            # If we haven't reached a Final State then keep playing
            if not self.current_node.state.is_over():
                
                # Player 1
                if self.current_node.state.current_player == 1:
                    if heuristic_1 is None:
                        new_node = player1()
                        self.current_node = new_node
                    else:
                        new_node = player1(heuristic_1)
                        self.current_node = new_node
    
                # Player 2
                else:
                    if heuristic_2 is None:
                        new_node = player2()
                        self.current_node = new_node
                    else:
                        new_node = player2(heuristic_2)
                        self.current_node = new_node

            # Found a Final State
            else:
                if (self.current_node.state.winner == 1):
                    self.write_winner(screen, player1.__name__)
                else: # (self.current_node.state.winner == 2)
                    self.write_winner(screen, player2.__name__)

            # Main Event Loop
            for event in pygame.event.get():
                # Close the App
                if (event.type == pygame.QUIT):
                    return 0
                
                # Reseting the game
                if (event.type == pygame.KEYDOWN):
                    if (event.key == pygame.K_r):
                        self.current_node = TreeNode(state=Connect_Four_State())
                 
            # Updates the Window
            pygame.display.update()

        # Went back to the Game Menu's
        return 1

    def run(self) -> None:
        # Initializing Window / Screen
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Connect-4")
        ICON_PATH = get_asset_path('Assets', 'Connect-Four.png')
        ICON_IMG = pygame.image.load(ICON_PATH).convert_alpha()
        pygame.display.set_icon(ICON_IMG)

        # Creating Main Menu Background Image
        BACKGROUND_PATH = get_asset_path('Assets', 'Connect-Four_MainMenu.png')
        BACKGROUND_IMG = pygame.image.load(BACKGROUND_PATH).convert_alpha()
        Main_Menu_Image = Image(BACKGROUND_IMG, -25, -25, 0.7)
        
        # Creating the Game Mode Sub Menu Background
        MODES_PATH = get_asset_path('Assets', 'Connect-Four_GameModes.png')
        MODES_IMG = pygame.image.load(MODES_PATH).convert_alpha()
        Modes_Image = Image(MODES_IMG, -25, -25, 0.7)

        # Creating Buttons
        BACK_PATH = get_asset_path('Assets', 'Back.png')
        BACK_IMG = pygame.image.load(BACK_PATH).convert_alpha()
        Back_Btn = Button(BACK_IMG, 20, 20, 0.1)
        
        START_PATH = get_asset_path('Assets', 'Start.png')
        START_IMG = pygame.image.load(START_PATH).convert_alpha()
        Start_Btn = Button(START_IMG, 260, 100, 0.3)
        
        RANDOM_PATH = get_asset_path('Assets', 'Random.png')
        RANDOM_IMG = pygame.image.load(RANDOM_PATH).convert_alpha()
        Random_Btn = Button(RANDOM_IMG, 40, 170, 0.2)
        
        A_STAR_PATH = get_asset_path('Assets', 'A_Star_Search.png')
        A_STAR_IMG = pygame.image.load(A_STAR_PATH).convert_alpha()
        A_Star_Btn = Button(A_STAR_IMG, 30, 350, 0.18)
        
        MINIMAX_PATH = get_asset_path('Assets', 'MiniMax.png')
        MINIMAX_IMG = pygame.image.load(MINIMAX_PATH).convert_alpha()
        MiniMax_Btn = Button(MINIMAX_IMG, 535, 170, 0.14)
        
        MCTS_PATH = get_asset_path('Assets', 'MCTS.png')
        MCTS_IMG = pygame.image.load(MCTS_PATH).convert_alpha()
        MCTS_Btn = Button(MCTS_IMG, 535, 350, 0.2)

        DT_PATH = get_asset_path('Assets', 'DecisionTree.png')
        DT_IMG = pygame.image.load(DT_PATH).convert_alpha()
        DT_Btn = Button(DT_IMG, 290, 500, 0.18)
        
        # Create a Flag to keep track of current state of the Application / GUI
        run = True

        # Main Loop
        while run:
            if (self.menu == "Main_Menu"):
                Main_Menu_Image.Display(screen)
                if (Start_Btn.Action(screen)):
                    self.menu = "Modes"

            if (self.menu == "Modes"):
                Modes_Image.Display(screen)
                self.write(font='Arial', text=" Game Modes ", size=50, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(200, 50), screen=screen)
                if (Back_Btn.Action(screen)):
                    self.menu = "Main_Menu"

                self.write(font='Arial', text=" Random ", size=25, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(40, 280), screen=screen)
                if (Random_Btn.Action(screen)):
                    self.menu = "Random"

                self.write(font='Arial', text=" A* Search ", size=25, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(33, 480), screen=screen)
                if (A_Star_Btn.Action(screen)):
                    self.menu = "A_Star"

                self.write(font='Arial', text=" MiniMax ", size=25, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(545, 300), screen=screen)
                if (MiniMax_Btn.Action(screen)):
                    self.menu = "MiniMax"

                self.write(font='Arial', text=" MCTS ", size=25, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(558, 485), screen=screen)
                if (MCTS_Btn.Action(screen)):
                    self.menu = "MCTS"

                self.write(font='Arial', text=" ID3 ", size=25, color=LIGHT_BLUE, bg_color=WHITE, bold=True, pos=(315, 600), screen=screen)
                if (DT_Btn.Action(screen)):
                    self.menu = "ID3"
                            
            if (self.menu == "Random"):
                if (self.run_game(screen=screen, player1=self.random, player2=self.player, heuristic_1=None, heuristic_2=None)):
                    self.menu = "Modes"
                else:
                    run = False
            
            if (self.menu == "A_Star"):
                if (self.run_game(screen=screen, player1=self.A_Star, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None)):
                    self.menu = "Modes"
                else:
                    run = False
            
            if (self.menu == "MiniMax"):
                if (self.run_game(screen=screen, player1=self.minimax, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None)):
                    self.menu = "Modes"
                else:
                    run = False
            
            if (self.menu == "MCTS"):
                if (self.run_game(screen=screen, player1=self.mcts, player2=self.player, heuristic_1=self.id3_heuristic_V2, heuristic_2=None)):
                    self.menu = "Modes"
                else:
                    run = False

            # ID3 Algorithm [According to the Database where the dataset's from the algorithm must be the first to play]
            if (self.menu == "ID3"):
                self.menu = "Modes"
            
            # Main Event Loop
            for event in pygame.event.get():
                if (event.type == pygame.QUIT):
                    run = False
            
            # Updates the Window
            pygame.display.update()
        pygame.quit()