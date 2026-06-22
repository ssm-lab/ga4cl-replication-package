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

[Experiment setup]()

[Results]()

## Content description

`01-`:

`02-data`: Contains experimental data produced in accordance with the experiment settings

`03-analysis`: Contains Python analysis scripts to obtain the results in the `04-results` folder

`04-results`: Contains the plots and statistical significance values that are used in the publication
