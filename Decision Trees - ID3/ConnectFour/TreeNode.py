from __future__ import annotations
import numpy as np
from .ConnectFourState import (Connect_Four_State)

class TreeNode:
    def __init__(self, state:Connect_Four_State, parent=None) -> None:
        # Stores a State of the Game
        self.state = state
        # Keeps a reference to his Parent Node
        self.parent = parent
        # Stores all the generated nodes
        self.children = []
        # Initializing the number of visits [Used in the MCTS]
        self.visits = 0
        # Defining a variable to keep track of the amount of wins after choosing a certain node
        self.wins = 0
        # Declaring a variable to store all moves that were not yet explored
        self.unexplored_actions = self.state.actions.copy()

    def is_root(self) -> bool:
        # Returns if the current node is the root
        return self.parent is None
    
    def is_leaf(self) -> bool:
        # Returns True if the Node does not have any children and therefore is a leaf
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        # Returns if the current node contains a terminal node or not
        return self.state.is_over()

    def fully_expanded(self) -> bool:
        # Returns True if a node is fully expanded
        return self.unexplored_actions.size == 0
    
    def pick_random_child(self) -> TreeNode:
        # Picking a random unexplored_action
        [ncol] = np.random.choice(self.unexplored_actions, size=1)
        # Creating a new node with the random action picked
        new_child = self.generate_new_node(ncol)
        # Returns the New Child
        return new_child
            
    def generate_new_node(self, ncol:int) -> TreeNode:
        # Creates a new state after the move
        new_state = self.state.move(ncol)
        # Wraps it with a TreeNode
        new_node = TreeNode(state=new_state, parent=self)
        # Inserts the New Node into the Current Node's Children
        self.children.append(new_node)
        # Updating the unexplored actions
        self.unexplored_actions = np.delete(self.unexplored_actions, np.where(self.unexplored_actions == ncol))
        # Returns the generated Node
        return new_node

    def read_state_node(self, file_path:str) -> TreeNode:
        # Reading a State
        new_state = Connect_Four_State().read_state(file_path)
        # Creating a new node based on the read state
        new_node = TreeNode(state=new_state,parent=None)
        # Returning the new node
        return new_node

    def __str__(self) -> str:
        return str(self.state)
    
    def __hash__(self) -> hash:
        return hash(str(self.state) + str(self.parent) + "".join([str(child) for child in self.children]))

    def __eq__(self, other:object) -> bool:
        if (not isinstance(other, TreeNode)):
            raise Exception(f"Sorry, other object is not an instance of {self.__class__.__name__}")
        return self.__hash__() == other.__hash__()
    
if __name__ == "__main__":
    initial_state = Connect_Four_State()
    node = TreeNode(initial_state)
    print(node.pick_random_child())