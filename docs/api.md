# Project Davis API Reference

**Author:** Onyerikam Godwin ([Web Essentz](https://github.com/WebEssentz))
**Parent Organization:** Avocado
**Beneficiary Organization:** Ossie

---

## 1. Overview

The `davis_ai` library is a self-contained cognitive engine. Its public API is designed to expose the primary entry points for training, evaluation, and inference, while abstracting away the internal complexities. The API is centered around a few key, high-level classes.

## 2. Core Engine: `davis_ai.engine`

This is the strategic heart of Davis, containing the "brain" and its "imagination."

### `DavisChessModel`
The neural network architecture. While it can be instantiated directly, the preferred method is to load it via its checkpointing methods.

*   `DavisChessModel(input_channels, num_residual_blocks, num_filters)`
    *   Constructor to create a new model instance. Usually only called once for the very first model.
*   `model.save_checkpoint(path: str)`
    *   Saves the model's configuration and learned weights to a file. This is the standard way to persist a trained brain.
*   `DavisChessModel.load_checkpoint(path: str) -> DavisChessModel`
    *   **[Primary Usage]** A static method to load a model from a file. This is the main entry point for using a pre-trained model.

### `InferenceEngine`
The sole gateway for getting predictions from a model. It handles device management (CPU/GPU) and ensures the model is in evaluation mode.

*   `InferenceEngine(model_path: str, device: str = None)`
    *   Constructor that takes the path to a saved model file and loads it into memory.
*   `engine.predict(board: Board) -> Tuple[torch.Tensor, float]`
    *   Takes a `Board` object and returns the model's raw evaluation: a policy tensor (probabilities for all 4672 moves) and a value prediction (a float from -1.0 to 1.0).

### `MonteCarloTreeSearch`
The imagination engine. It uses an `InferenceEngine` to explore possibilities.

*   `MonteCarloTreeSearch(inference_engine: InferenceEngine, config: dict)`
    *   Constructor. Requires a live `InferenceEngine` and a configuration dictionary for its parameters (c_puct, dirichlet_alpha, etc.).
*   `mcts.search(board: Board, num_simulations: int) -> MCTSNode`
    *   The primary search function. Takes a `Board` and a number of simulations to run. It returns the root `MCTSNode`, which contains the aggregated statistics of the entire search.
*   `mcts.select_move(root: MCTSNode, temperature: float) -> chess.Move`
    *   Takes the root node from a completed search and selects a final move based on the visit counts and the given temperature. `temperature=0` is deterministic (best move), `temperature>0` is stochastic (exploratory).

## 3. Training Loop: `davis_ai.training`

These modules provide the functions that power the generational learning cycle.

### `run_self_play_and_save(model_path, save_dir, config)`
*   A high-level function that initializes a `SelfPlayWorker`, orchestrates a full game using the specified model, and saves the resulting `GameRecord` object to a file.

### `analyze_and_enhance_game(game_record, analyst_engine, config)`
*   Implements our **Echo Chamber Reinforcement (ECR)** algorithm. Takes a `GameRecord` and an `InferenceEngine` to act as the "Analyst." It returns a list of processed training samples, including the powerful counterfactual data.

### `run_training_cycle(model_path, processed_data_path, new_model_save_path, config)`
*   The main training function. It loads a base model, creates a `DataLoader` from the processed game data, runs the training for a configured number of epochs, and saves the resulting new model.

### `run_evaluation(candidate_path, best_path, config)`
*   The tournament function. It loads two models, pits them against each other for a set number of games, and returns `True` if the candidate model is demonstrably superior based on the configured win threshold.

## 4. Master Controller: `davis_ai.correction`

This contains the master class that uses all other API components.

### `ReinforcementLearner`
The orchestrator of the entire end-to-end process.

*   `ReinforcementLearner(config: dict)`
    *   Initializes the master controller. Requires the global `DAVIS_CONFIG` dictionary. It sets up all necessary directories.
*   `learner.run_loop()`
    *   **[Primary Usage]** This is the main entry point to the entire system. Calling this method begins the generational cycle of self-play, analysis, training, and evaluation, continuing until the configured number of generations is complete. This is the "ignition switch" called by `scripts/train_davis.py`.

---
*This API provides the building blocks for Ossie's intelligent systems. Treat it as the foundational layer upon which all future logic will be built.*
