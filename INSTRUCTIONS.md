# AURORA Project Instructions

## Project Structure
The project has been reorganized for clarity:

- **`aurora/`**: Core package containing all source code.
    - **`controller.py`**: Main logic for the Meta-Evolutionary Controller.
    - **`data_loader.py`**: Utility to load Google/Azure traces and generate synthetic data.
    - **`main.py`**: Entry point for running training and evaluation.
    - **`config/`**: Configuration files and generated parameters (e.g., `best_ppo_params.json`).
    - **`evaluation/`**: Scripts for evaluating explainability and performance.
    - **`simulators/`**: Realistic resource simulation logic.
    - **`tests/`**: Unit tests for the system.
- **`data/`**: Large datasets (Google traces) and generated test sets.
- **`results/`**: Output directory for evaluation results, logs, and artifacts (e.g., `explainability_samples.json`, plots).
- **`cluster-data-2019/`**: Specific subset of Google Trace data.

## How to Run

### 1. Training the Main Solution
To train the Meta-Evolutionary Controller (the "Attacker-Defender" loop):

```bash
python -m aurora.main --mode train --episodes 10
```
*   **--episodes**: Number of self-play iterations to run.
*   The system will automatically use the local Google 2019 traces if configured in `data_loader.py`.

### 2. Tuning Baselines (PPO)
To optimize hyperparameters for the PPO baseline:

```bash
python -m aurora.tune_hyperparameters
```
*   This will save the best parameters to `aurora/config/best_ppo_params.json`.

### 3. Evaluating Baselines
To run the optimization solver (OR-Tools) baseline:

```bash
python -m aurora.optimization_solver
```

### 4. Running Comprehensive Evaluation
To evaluate the trained solution against baselines using the stratified test set:

```bash
python -m aurora.main --mode evaluate
```
*   This uses the test cases in `data/test_set.json`.

### 5. Evaluating Explainability
To generate policy explanation samples and simulate human evaluation:

```bash
python -m aurora.evaluation.explainability_evaluator
```
*   Results will be saved to `results/explainability_samples.json`.

### 6. Running Tests
To verify system components:

```bash
pytest aurora/tests/test_realistic_simulator.py
```

## Navigation Tips
- All core logic is inside `aurora/`.
- Check `results/` for outputs after running scripts.
- Use `aurora.main` as the primary interface.
