# Replication package
**For the paper _A Model-Driven Approach for Developing Families of Reinforcement Learning Environments_.**

(Accepted for [MODELS 2026](https://conf.researchr.org/home/models-2026).)

## About
Virtual training environments are software-intensive systems in which reinforcement learning (RL) agents learn, adapt, and demonstrate meaningful behavior. Virtual training environments offer a safe and cost-efficient alternative to training agents in real-world settings. However, to converge, most realistic RL problems require training in multiple, mostly similar but slightly different environments---i.e., families of environment variants. The typical development process of environment families is a labor-intensive and error-prone manual endeavor that does not scale well. To alleviate these issues, in this paper, we propose a model-driven approach for developing families of RL training environments.
To obtain the family of environments, we develop an approach and prototype tool. In our approach, a hybrid genetic algorithm---a combination of population-based global search and heuristic local search---generates environment families. Mutations and constraints are expressed as model transformations and are operationalized into a search process by a state-of-the-art model transformation engine.
We demonstrate the soundness of our approach in a wildfire mitigation scenario and curriculum learning---a particular learning paradigm that relies on environment families.

## Table of contents
[Content description]()

[Reproduction of analysis]()

[Reproduction of experimental data]()

[Experiment settings]()

[Results]()

## Content description

`01-configurations`: contains the generated environment configurations
- `environment-config.json`: the training environment configuration
- `learning-config.json`: the RL configuration

`02-train-data`: contains trained agents' data produced by running the scripts in `03-scripts`
- `<experiment>/start_stage_1/` … `start_stage_5/`: final checkpoints (`final_env_N.npy`) and periodic evaluation metrics for each curriculum run
- `<experiment>/baseline/<label>/`: equivalent checkpoint and eval data for each baseline agent

`03-scripts`: contains Python scripts to reproduce the training results and plots in `04-results`

`04-results`: contains the plots and visualizations used in the publication
- `map_visualization/`: visualization of each environment in the generated curriculum
- `plots/`: cumulative reward plots during training
- `statistics/`: evaluation results


## Reproduction of analysis

### Requirements

Python 3.11 or higher is required. Set up a virtual environment from the `03-scripts` directory:

```bash
cd 03-scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 1 — Generate map visualizations

```bash
python map_visualizer.py
```

Saves the visualization of each environment to `04-results/map_visualization/`.

### Step 2 — Training

```bash
python runner.py --all-stages --baseline all
```


This will:
1. Build the full curriculum (E1 → E6) ordered by environment complexity
2. Run 5 partial curricula starting from E1, E2, E3, E4, and E5 through the target environment (E6)
3. Run a direct-training baseline for each environment
4. Save agent checkpoints to `02-train-data/`
5. Save the plots to `04-results/plots/`.



### Step 3 — Evaluation

```bash
python evaluator.py
```

Reads saved checkpoints from `02-train-data/` and saves `evaluation_results.csv` to `04-results/statistics/`.

## Reproduction of experimental data

TODO: Istvan for Eclipse part
Question: Table 3 about generation time?

## Experiment settings
### Problem
The target environment used in the experiments:

<img width="400" height="400" alt="env_6-complexity_0 93" src="https://github.com/user-attachments/assets/c8e799f5-e4ac-45b4-9bab-fcd62a72fc2d"  />


### Settings and hyperparameters
| Parameter | Value |
|---|---|
|Mutation rate | 0.85 |
|Complexity measure | The number of feasible paths between the start and the goal  |
|Diversity measure |  Shannon entropy |
|Population size | 6 |
|Training budget per environment | 50,000 steps |
|RL method |  Q-learning |
|Learning rate (α) | 0.1 |
|Discount factor (γ) | 0.99 |

## Results

### Generated curricula
(1) <img width="300" height="300" alt="env_1-complexity_0 00" src="https://github.com/user-attachments/assets/47e18294-bcce-42b4-8a41-05a2f5740a80" />
(2) <img width="300" height="300" alt="env_2-complexity_0 20" src="https://github.com/user-attachments/assets/bc68702d-4386-45c9-a30b-4aeaf7ff1c87" />
(3) <img width="300" height="300" alt="env_3-complexity_0 48" src="https://github.com/user-attachments/assets/0765aaf0-4fa1-44d8-ac6a-4574707a7647" />

(4) <img width="300" height="300" alt="env_4-complexity_0 59" src="https://github.com/user-attachments/assets/44b7c98f-2130-4289-985e-424ee1aaf12e" />
(5) <img width="300" height="300" alt="env_5-complexity_0 67" src="https://github.com/user-attachments/assets/4956659b-433b-4131-bfe0-deda8f8deeb0" />
(6) <img width="300" height="300" alt="env_6-complexity_0 93" src="https://github.com/user-attachments/assets/8764b390-df4e-439f-b10e-38a91543bd1f" />

### Cumulative reward during training

**Cumulative reward across all curricula**

<img width="480" height="300" alt="cumulative_reward_all-v2" src="https://github.com/user-attachments/assets/bff9b8e8-d6ca-4c4c-9601-abf854e6ec14" />

**Cumulative reward with different prefixes**

<img width="480" height="300" alt="cumulative_reward_start_1-v2" src="https://github.com/user-attachments/assets/267f4708-0dea-44b4-a1ef-7cc338190a40" />
<img width="480" height="300" alt="cumulative_reward_start_2-v2" src="https://github.com/user-attachments/assets/2d0ebbe1-3c13-4bbe-9f65-0669ea63045c" />
<img width="480" height="300" alt="cumulative_reward_start_3-v2" src="https://github.com/user-attachments/assets/74c4a9f9-f6de-4476-95ec-77752466c5b4" />
<img width="480" height="300" alt="cumulative_reward_start_4-v2" src="https://github.com/user-attachments/assets/54f26b65-1fd6-42b0-ae4c-5720a42cb302" />
<img width="480" height="300" alt="cumulative_reward_start_5-v2" src="https://github.com/user-attachments/assets/6cbebade-6653-4b68-b946-18d1c3e4db6e" />





