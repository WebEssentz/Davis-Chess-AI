# The Architecture of Project Davis: A Metacognitive Loop

---
## 1. Design Philosophy

The architecture of Davis is not merely a pipeline of modules; it is an attempt to model a cognitive process. We separate the system into components that mirror distinct mental faculties: raw intuition, focused imagination, long-term memory, and a mechanism for critical self-reflection. Our goal was to build a system that could "think about its own thinking."

## 2. Core Components

The system is best understood as two interacting loops: the **Performance Loop** (a single move) and the **Learning Loop** (a full generation).

### 2.1. The Performance Loop (Making a Move)

This loop executes for every turn in every game.

```
+----------------+      1. Request      +--------------------+
|   Self-Play    |       Prediction     |  Inference Engine  |
|    Worker      |--------------------->|  (Nervous System)  |
+----------------+                      +--------------------+
        ^                                         | 2. Tensor In
        | 6. Return Best Move                     | 3. Prediction Out
        |                                         v
+----------------+      5. Evaluate      +--------------------+
|      MCTS      | <-------------------- | DavisChessModel    |
| (Imagination)  |      4. Policy/Value |   (The Brain)      |
+----------------+                      +--------------------+
```

1.  **The Game Worker** asks for the best move from a given `Board` state.
2.  It uses the **Monte Carlo Tree Search (MCTS)** engine to simulate thousands of possible futures.
3.  For each new, imagined position in the search tree, the MCTS asks the **Inference Engine** for a "gut feeling" about that position.
4.  The Inference Engine feeds the board tensor to the **`DavisChessModel`** (our ResNet-like neural network), which returns a **policy** (a guess about the best moves) and a **value** (a guess about who is winning).
5.  This data guides the MCTS search to focus on more promising futures.
6.  After thousands of simulations, the MCTS search returns the most robust move back to the Game Worker.

### 2.2. The Learning Loop (Becoming Smarter)

This is the generational loop orchestrated by the `ReinforcementLearner`.

```
          +-------------------------------------------------------------+
          |                                                             |
+----------------------+      1. Play      +-------------------------+  | 5. Update
|  Current Best Model  |------------------>|  Self-Play Games        |  |  Best Model
| (data/models/best)   |                   |  (data/games/gen_N)     |  |
+----------------------+                   +-------------------------+  |
          ^                                           |                |
          |                                           | 2. Analyze     |
          | 4. Evaluate & Promote                     | (ECR)          |
          |                                           v                |
+----------------------+      3. Train     +-------------------------+  |
|  Candidate Model     |<------------------| Processed Training Data |  |
| (data/models/cand)   |                   | (data/processed/gen_N)  |  |
+----------------------+                   +-------------------------+  |
          |                                                             |
          +-------------------------------------------------------------+
```

## 3. The Core Innovation: Echo Chamber Reinforcement (ECR)

The critical step in our architecture is **Step 2 (Analyze)**. This is where we break from the standard AlphaZero paradigm.

**The Problem:** A self-play agent can develop flawed but effective strategies against itself. If it never explores a certain opening, it will never learn its weaknesses. This is a local maximum—an "echo chamber."

**Our Solution: The ECR Analyst.**

After a game is complete, we do not immediately use it for training. Instead:
1.  **Identify the Winner:** The game record is analyzed to determine the winning color.
2.  **Calculate Regret:** For every move the winner made, the Analyst performs a **much deeper MCTS search** than the one used during the game. It compares the model's initial, "quick" evaluation of the position with the value from this new, "deep" search. The difference is the **Regret**.
3.  **Find the Hidden Blunder:** The Analyst identifies the single move by the winner that had the highest Regret. This is the move where the winner was most "lucky"—the moment they made a suboptimal choice that their opponent failed to punish.
4.  **Generate a Counterfactual:** A new training sample is created for this "blunder" position. The input is the board state, but the output `policy` target is replaced. Instead of the visit counts from the original, flawed search, we use the visit counts from the new, deeper, more "correct" search.
5.  **Enhance the Dataset:** This counterfactual "lesson" is added to the training data.

**The Result:** ECR forces the model to learn from an objective, almost omniscient critic. It learns not just what leads to a win, but what the *most optimal path to that win* looked like. It is a mechanism designed to force the AI out of its own echo chamber and toward a more universal, robust understanding of the game.
```

---