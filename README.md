CL4AD
========

![Python version](https://img.shields.io/badge/Python-3.11-blue)

The first integration of automated curriculum learning into [GPUDrive](https://github.com/Emerge-Lab/gpudrive), a batched autonomous driving simulator . We implement an unsupervised environment design algorithm called Prioritized Level Replay (PLR), and combine it with utility functions based on regret, success and realism of the self-play RL agent's behavior. 

## Highlights

- CL4AD consists of three strategies for sampling scenarios from real driving datasets to train RL agents:
  - Default: GPUDrive's default sampling mechanism
  - Domain randomization: Uniformly randomly sampling scenarios
  - PLR: A well-known UED method that curates scenarios based on their utilities.
- CL4AD provides seven utility functions grouped into three categories:
  - **[Regret]** Average magnitude of generalized advantage estimate (AMGAE)
  - **[Regret]** Positive value loss (PVL)
  - **[Regret]** Maximum Monte Carlo (MaxMC)
  - **[Success]** Learnability (Learn)
  - **[Success]** Learnability-Hard (Learn-Hard)
  - **[Realism]** Goal-conditioned average distance error (GC-ADE)
  - **[Realism]** Action mean absolute error (Act-MAE)
- CL4AD currently operates following the PPO implementation via [Pufferlib](https://puffer.ai/), which was readily available in GPUDrive.
- CL4AD sequences traffic scenarios from real self-driving datasets, such as [Waymo Open Motion Dataset](https://github.com/waymo-research/waymo-open-dataset).
- We provide examples for how to configure curriculum learning strategies to train self-play agents in GPUDrive.

## Installation

Please follow instructions provided under the installation section in [GPUDrive](https://github.com/Emerge-Lab/gpudrive) repository.

## Usage

To configure curriculum learning strategies in CL4AD, the following parameters in the training configuration file should be set:
```
  ### Scenario sampling ###
  curriculum_type: plr # Options: default, domain_randomization (resample_scenes should be true)
  lambd: 0.95 # PLR: Discount factor for GAE
  buffer_size: 100000 # PLR: Replay buffer size. We set this to the size of the training dataset
  replay_rate: 0.5 # PLR: Probability of sampling unseen scenarios rather than replaying from the buffer.
  beta: 1.0 #0.1 # PLR: Temperature parameter for the score distribution
  rho: 0.1 # PLR: Staleness coefficient
  score_type: amgae # PLR score types: amgae, pvl, maxmc, learn, learn-goal, gc-ade, act-mae
  warm_up_steps: 0 # Number of warmup steps before sampling from the curriculum
  resample_scenes: true # Set to true for DR and PLR. Default may train with fixed scenarios.
  resample_dataset_size: 1_000 # Number of scenes to use in the dataset
  resample_interval: 2_000_000 $ Number of interactions to sample new scenarios 
  sample_with_replacement: true # Ignored except Default
  shuffle_dataset: false
```

We provide examples in `baselines/ppo/config`:

- `ppo_base_puffer_cl4ad.yaml`
- `ppo_base_puffer_cl4ad_computeab.yaml`

An implementation in self-play PPO provided in `baselines/ppo/ppo_pufferlib_cl4ad.py`.

## How does CL4AD operate?

We implement curriculum strategies under `gpudrive/cl4ad`:
- `default_curriculum.py`: Default sampling
- `domain_randomization.py`: DR
- `priortized_level_replay.py`: PLR

Each strategy essentially need a `sample()` and `update()` function, where the former is used to sample new traffic scenarios from the dataset and the latter is used to update the curriculum via the rollouts of the self-play agent.

## CL4AD in GPUDrive

Integration of CL4AD in GPUDrive, and in any other batched AD simulator, fundamentally requires three changes:
- Sending data from traffic scenarios, such as timesteps, vehicle_IDs, rewards, etc., to curriculum instance via `curriculum.update()`
- Setting scenarios based on the output of `curriculum.sample()`
- A set of traffic scenarios, currently implemented as a dataloader instance, to sample from.

We made changes to the following scripts under `gpudrive` package to enable this integration:
- `gpudrive/integrations/puffer/ppo.py`
- `gpudrive/env/env_puffer.py`
- `gpudrive/env/env_torch.py`