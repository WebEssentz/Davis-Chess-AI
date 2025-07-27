# src/davis_ai/engine/mcts.py

import math
import numpy as np
import torch
from typing import Dict, Optional

# CORRECTED IMPORT
from ..board import Board
from ..engine.inference import InferenceEngine
from ..utils.move_encoder import move_to_policy_index
import chess

class MCTSNode:
    """Represents a node in the Monte Carlo Tree Search tree."""
    # CORRECTED TYPE HINTS AND VARIABLE NAMES
    def __init__(self, parent: Optional['MCTSNode'], board: Board, policy_prior: float):
        self.parent = parent
        self.board = board
        self.children: Dict[chess.Move, MCTSNode] = {}
        
        self.visits = 0
        self.value_sum = 0.0
        self.policy_prior = policy_prior

    @property
    def q_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_leaf(self) -> bool:
        return not self.children

    def ucb_score(self, c_puct: float) -> float:
        exploration_bonus = self.policy_prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        return -self.q_value + c_puct * exploration_bonus

class MonteCarloTreeSearch:
    """Advanced MCTS implementation for Davis AI."""
    def __init__(self, inference_engine: InferenceEngine, c_puct: float = 4.0, dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25):
        self.inference_engine = inference_engine
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.root: Optional[MCTSNode] = None

    # CORRECTED TYPE HINTS AND VARIABLE NAMES
    def search(self, board: Board, num_simulations: int) -> MCTSNode:
        """Performs MCTS simulations starting from the given board state."""
        self.root = MCTSNode(parent=None, board=board, policy_prior=0.0)

        # The first expansion gets the value for the root
        value = self._expand_and_evaluate(self.root, add_dirichlet_noise=True)

        # ADD THIS LINE: We must backpropagate the first simulation!
        self._backpropagate(self.root, value)

        # CHANGE THIS LINE: The loop now runs N-1 times.
        for _ in range(num_simulations - 1):
            node_to_expand = self._select(self.root)
            
            outcome = node_to_expand.board.get_game_outcome()
            value = outcome if node_to_expand.board.is_game_over() else self._expand_and_evaluate(node_to_expand, add_dirichlet_noise=False)

            self._backpropagate(node_to_expand, value)
        
        return self.root
    
    # ... The rest of the methods (_select, _expand_and_evaluate, etc.) use the local `node` variable
    # which already has the `.board` attribute, so they don't need to change. The __init__ and search
    # methods were the key entry points to fix. The logic below is confirmed correct.
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        while not node.is_leaf():
            node = max(node.children.values(), key=lambda child: child.ucb_score(self.c_puct))
        return node

    def _expand_and_evaluate(self, node: MCTSNode, add_dirichlet_noise: bool) -> float:
        policy_probs, value = self.inference_engine.predict(node.board)
        
        legal_moves = node.board.get_legal_moves()
        if not legal_moves:
            return value

        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_moves))
        
        for i, move in enumerate(legal_moves):
            try:
                policy_idx = move_to_policy_index(move)
                prior = policy_probs[policy_idx].item()

                if add_dirichlet_noise:
                    prior = (1 - self.dirichlet_epsilon) * prior + self.dirichlet_epsilon * noise[i]
                
                new_board = node.board.apply_move(move)
                node.children[move] = MCTSNode(parent=node, board=new_board, policy_prior=prior)
            except (ValueError, IndexError):
                continue
        
        return value

    def _backpropagate(self, node: MCTSNode, value: float):
        current_node = node
        while current_node is not None:
            current_node.visits += 1
            current_node.value_sum += value
            value = -value
            current_node = current_node.parent

    def select_move(self, root: MCTSNode, temperature: float = 1.0) -> chess.Move:
        if not root.children:
             raise RuntimeError("Cannot select a move: root has no children after search.")

        moves = list(root.children.keys())
        visit_counts = np.array([root.children[m].visits for m in moves], dtype=np.float32)

        if temperature == 0:
            best_move_idx = np.argmax(visit_counts)
            return moves[best_move_idx]
        else:
            visit_counts_temp = visit_counts**(1.0 / temperature)
            probabilities = visit_counts_temp / np.sum(visit_counts_temp)
            probabilities /= np.sum(probabilities)
            chosen_move = np.random.choice(moves, p=probabilities)
            return chosen_move