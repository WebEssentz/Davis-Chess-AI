# Project Davis Training and Tuning Guide

**Author:** Onyerikam Godwin ([Web Essentz](https://github.com/WebEssentz))
**Parent Organization:** Avocado
**Beneficiary Organization:** Ossie

---

## 1. The Philosophy of Training Davis

Training Davis is not like training a standard machine learning model. You are not feeding it a static dataset. You are curating the life experiences of a learning entity. The goal is to create a virtuous cycle: better models play higher-quality games, which generates better training data, which in turn creates better models.

Your role as the architect is to tune the parameters of this cycle. The primary configuration file for all training operations is `src/davis_ai/config/settings.py`.

## 2. The Training Process: From Zero to Hero

To begin training, ensure your environment is set up and tested (`pip install -e .` and `pytest`). Then, execute the master script from the project root:

```bash
python scripts/train_davis.py
```

This script will:
1.  Check for a `best_model.pth` in `data/models/`. If one does not exist, it will create a new one with random weights. This is the "Generation 0" model.
2.  Begin the main `run_loop` for the number of generations specified in `DAVIS_CONFIG`.

You can monitor the entire process in real-time from the console and review the complete history in `logs/training_run.log`.

## 3. Tuning Key Hyperparameters

The art of training Davis lies in balancing the trade-offs between the quality of self-play data and the speed of generating it. Below are the most critical parameters to tune in `settings.py`.

### Self-Play and Search Quality

*   `simulations_per_move`: **This is the most important parameter for data quality.** It controls how deeply the AI "thinks" about each move during self-play.
    *   **Low values (50-200):** Games will be generated very quickly, but the moves will be of poor quality, especially in the early stages. The AI may learn bad habits.
    *   **High values (800-1600+):** Produces very high-quality, human-like games. Learning will be much more efficient *per game*, but the overall training process will be significantly slower.
    *   **Recommendation:** Start with `400` and increase as your hardware allows.

*   `mcts_c_puct`: Controls the exploration vs. exploitation trade-off in the MCTS search. Higher values encourage the AI to explore more novel but potentially worse moves. It is generally safe to leave this between `1.5` and `5.0`.

### Generational Cycle

*   `games_per_generation`: The number of self-play games to generate before a new model is trained.
    *   Too few games, and the new model won't have enough data to learn meaningful patterns.
    *   Too many games, and the training cycles will be very infrequent.
    *   **Recommendation:** Aim for at least 1,000-5,000 games per generation. A good starting point is `2048`.

*   `training_epochs`: The number of times the trainer will loop over the entire dataset of new games.
    *   **Recommendation:** A high number is not always better and can lead to overfitting on the most recent batch of games. Values between `5` and `15` are typical.

### Echo Chamber Reinforcement (ECR)

*   `ecr_analyst_simulations`: How deeply the "Analyst" thinks when searching for hidden blunders.
    *   **Rule of Thumb:** This should always be significantly higher than `simulations_per_move`. A 2x-4x multiplier is a good starting point. If `simulations_per_move` is 400, setting this to `800` or `1600` ensures the Analyst is genuinely "smarter" than the player was.

### Evaluation

*   `eval_win_threshold`: The gatekeeper for progress. This prevents the AI from getting worse due to a lucky training run.
    *   `0.51` (51%): Promotes very small improvements, leading to rapid but potentially unstable growth.
    *   `0.55` (55%): The standard used by DeepMind. Ensures that a new model is demonstrably and significantly better before it becomes the new champion. This is the recommended setting.

## 4. Hardware Considerations

This project is computationally intensive.
*   **CPU:** Training is possible on a modern multi-core CPU, but it will be very slow. A single generation might take days.
*   **GPU:** A CUDA-enabled NVIDIA GPU is **highly recommended**. It will accelerate the neural network forward passes, which are the primary bottleneck in both MCTS and the training phase. Performance gains can be 10x-100x over a CPU.

Training Davis is a marathon, not a sprint. Be patient. Observe its games. Tune its parameters. You are not just running a program; you are guiding the evolution of an intelligence.