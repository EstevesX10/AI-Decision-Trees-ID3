import heapq
from typing import (Callable)
from .ConnectFourState import (Connect_Four_State)
from .TreeNode import (TreeNode)
from math import (sqrt, log)
from time import (time)

'''
# --------- #
| A* Search |
# --------- #
'''

def A_Star_Search(initial_node:TreeNode, heuristic:Callable[[TreeNode], int]) -> TreeNode:

    # Setting a method in the TreeNode Class - Compares 2 Nodes taking into consideration their parent state's heuristic as well as the respective path's cost
    setattr(TreeNode, "__lt__", lambda self, other: ((heuristic(self.parent.state) - len(self.parent.state.move_history) + 1)) < (heuristic(other.parent.state) - len(other.parent.state.move_history) + 1))
    
    # Setting the Initial Node
    root = initial_node
    # Initializing a queue to help manage the generated nodes
    queue = [root]
    # Creating a set of visited_states so that we don't waste time generating new_states from an already visited state
    visited_states = set()

    # While we have nodes inside the queue
    while queue:
        
        # Pop current_node [Using a Max Heap]
        current_node = heapq._heappop_max(queue)

        # Continue if the state was already visited
        if current_node.state in visited_states:
            continue

        # Updating the visited_states set
        visited_states.add(current_node.state)

        # Checking if we found a Final State [if so return it]
        if current_node.state.is_over():
            
            # Finding next node after the "current_node" inside the "final_node"'s nth parents
            while current_node.parent != initial_node:
                current_node = current_node.parent

            # Returning the best next node
            return current_node

        # Generating new_states and adding them to the queue (wrapped with a TreeNode) if they were not visited
        for _, new_state in current_node.state.generate_new_states():
            if (new_state not in visited_states):
                child = TreeNode(state=new_state, parent=current_node)
                heapq.heappush(queue, child)
    
    # If we didn't found a Solution then we return None
    return None

'''
# ------- #
| MiniMax |
# ------- #
'''

def MiniMax_Move(state:Connect_Four_State, depth:int, alpha, beta, maximizing:bool, player:int, evaluate_func:Callable[[TreeNode], int]) -> float:
    """ MinMax with Alpha-Beta Pruning - EXTRA """
    
    # Reached the root [depth = 0] or found a Final State
    if depth == 0 or state.winner != -1:
        return abs(evaluate_func(state)) * (1 if player == 2 else -1)
    
    # Current layer is focused on Maximizing
    if maximizing:
        max_eval = float('-inf')
        for ncol in state.actions:
            new_state = state.move(ncol)
            eval = MiniMax_Move(new_state, depth - 1, alpha, beta, False, 3 - player, evaluate_func)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
        
    # Current layer is focused on Minimizing
    else:
        min_eval = float('+inf')
        for ncol in state.actions:
            new_state = state.move(ncol)
            eval = MiniMax_Move(new_state, depth - 1, alpha, beta, True, 3 - player, evaluate_func)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def execute_minimax_move(evaluate_func:Callable[[TreeNode], int], depth:int) -> TreeNode:
    def execute_minimax_move_aux(current_node):
        # Initializing the best move and evaluation parameters
        best_move = None
        best_eval = float('-inf')
        
        # Looping through all possible moves and evaluating each new state [using the minmax algorithm]
        # If they are better than the current best then they replace it
        for ncol in current_node.state.actions:
            new_state = current_node.state.move(ncol)
            new_state_eval = MiniMax_Move(new_state, depth - 1, float('-inf'), float('+inf'), False, current_node.state.current_player, evaluate_func)
            if new_state_eval > best_eval:
                best_move = new_state
                best_eval = new_state_eval
        new_node = TreeNode(state=best_move, parent=current_node)
        return new_node
        
    return execute_minimax_move_aux

def MiniMax(node:TreeNode, heuristic:Callable[[TreeNode], int], depth_search=5) -> TreeNode:
    # Executing a MiniMax move with a depth search given
    return execute_minimax_move(heuristic, depth_search)(node)

'''
# ----------------------- #
| Monte Carlo Tree Search |
# ----------------------- #
'''

def uct(node:TreeNode) -> float: 
    # Unvisited Nodes
    if (node.visits == 0):
        return float('+inf')

    # Upper Confidence Bound Applied to Trees to evaluate each branch
    return (node.wins / (node.visits + 1)) + (sqrt(2* log(node.parent.visits) / (node.visits + 1)))

def best_uct(node:TreeNode) -> float:
    # Returns the node's child with the highest uct value
    return max(node.children, key=lambda n: uct(n))
    
def Expansion(node:TreeNode, heuristic:Callable[[TreeNode], int]) -> TreeNode: # Initially the node is the root
    
    # Looking for a non fully expanded node
    while node.fully_expanded():
        node = best_uct(node)
    
    # Found a Terminal Node
    if node.is_terminal():
        return node
    
    # Evaluating the Scores [based on the Heuristic] for each unexplored move and returning the best one
    _, best_ncol = max((heuristic(node.state.move(col)), col) for col in node.unexplored_actions)
    child = node.generate_new_node(best_ncol)

    return child

def rollout_policy(node:TreeNode) -> TreeNode:
    # Applying a Rollout Policy -> in this case: Random Moves
    return node.pick_random_child()

def Rollout(node:TreeNode) -> int: # Also called Simulation
    # Saving a link to the initial node
    initial_node = node

    # Simulating a game using only random moves [until we find a terminal board]
    while not node.is_terminal():
        node = rollout_policy(node)

    # Updating the initial node's chldren since we are only doing Simulations
    initial_node.children = []
    
    # Returns the Winner
    return node.state.winner

def update_stats(node:TreeNode, winner:int) -> None:
    # Updating the Node's visits and the amount of Win's reached
    node.visits += 1

    # Checking for a Tie
    if (winner == 0):
        return

    # Checking if the previous player is the winner since we are trying to evaluate his choice upon the possible moves (described as children)
    if winner == node.state.previous_player():
        node.wins += 1
    
def Backpropagation(node:TreeNode, winner:int) -> None:
    # Updating the Node upon the Discovered Results
    update_stats(node, winner)
    
    # Base Case - When we reach the root we must stop
    if node.is_root():
        return
    
    # Updating the Parent Node
    Backpropagation(node.parent, winner)

def pick_best_child(node:TreeNode) -> TreeNode:
    # Selecting the best child [The one that was visited the most]
    best_node = max(node.children, key=lambda n: n.visits)

    # Since after exploring we are left with no possible actions in the unexplored_actions,
    # we must reset them so that the next player can perform a valid action
    best_node.unexplored_actions = best_node.state.actions.copy()
    
    # Returning best node according to the heuristic
    return best_node

def resources_left(start_time:time) -> bool:
    # Creating a Function that determines when to stop the MCTS Algorithm
    TIME_TO_TRAIN = 5.0
    return (time() - start_time) < TIME_TO_TRAIN

def MCTS(root:TreeNode, heuristic:Callable[[TreeNode], int]) -> TreeNode:
    # Saving the Initial Instant
    start = time()
    # Executing the Algorithm while there are resources left
    while(resources_left(start)):
        
        # Performs Expansion
        leaf = Expansion(root, heuristic)
        
        # Performs Rollout
        Simulation_Winner = Rollout(leaf)
        
        # Performs Backpropagation
        Backpropagation(leaf, Simulation_Winner)
        
    # Returns the Best child based on the number of visits
    return pick_best_child(root)