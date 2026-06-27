# Artifact Status

We apply for the following ACM badges for the paper
**_A Model-Driven Approach for Developing Families of Reinforcement Learning Environments_**:

## Artifact Evaluated — Functional

- **Documented**: `README.md` explains the artifact's purpose, folder structure, and step-by-step reproduction instructions. `REQUIREMENTS.md` specifies the software environment. All Python files include module and function level docstrings.
- **Consistent**: The artifact matches the paper's experiments. Configs in `01-configurations/` match the paper's hyperparameter table. Results in `04-results/` match the paper's figures and statistics.
- **Complete**: All scripts, configs, and pre-generated environment sequences needed to reproduce results are included. `04-results/` contains the reported plots and evaluation CSVs.
- **Exercisable**: Each reproduction step runs with a single command from `03-scripts/`.
- **Verification and validation**: `evaluator.py` evaluates trained agents and produces `evaluation_results.csv` as quantitative evidence of the results.

## Artifact Evaluated — Reusable

- **Well-structured**: A numbered folder convention (`01-configurations`, `02-train-data`, `03-scripts`, `04-results`) separates inputs, intermediate data, scripts, and outputs.
- **Carefully documented**: Scripts have module and function level docstrings. `README.md` covers content and reproduction steps in full.
- **Standard formats**: Configs use JSON, results use CSV and PNG. Dependencies are managed via `requirements.txt` and a virtual environment.
- **Reusable**: `generate_random_config.py` and the training infrastructure (`runner.py`, `trainer.py`, `run_comparison.py`) work with new environment configs and learning parameters, letting researchers apply the pipeline to other problems.

## Artifact Available

- Publicly available on GitHub and archived on Zenodo.
- Released under the MIT License.
- DOI and Zenodo link provided in `README.md`.
