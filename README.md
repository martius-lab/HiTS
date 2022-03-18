# Hierarchical reinforcement learning with Timed Subgoals (HiTS)

This repository contains code for reproducing experiments from [our paper "Hierarchical reinforcement learning with Timed Subgoals"](https://proceedings.neurips.cc/paper/2021/hash/b59c21a078fde074a6750e91ed19fb21-Abstract.html). The implementation of the *Hierarchical reinforcement learning with Timed Subgoals* (HiTS) algorithm can be found in the [Graph-RL repository](https://github.com/nicoguertler/graph_rl). A short [video presentation](https://youtu.be/JkPaI3uZU6c) summarizes the algorithm as well as our experiments.

HiTS enables sample-efficient learning in sparse-reward, long-horizong tasks. In particular, it extends subgoal-based hierarchical reinforcement learning to environments with dynamic elements which are, most of the time, beyond the control of the agent. Due to the use of timed subgoals and hindsight action relabeling the higher level sees transitions that are consistent with a stationary effective environment. As a result both levels in the hierarchy can learn concurrently and efficiently.

The three benchmark tasks in dynamic environments from the paper are contained in the [dynamic-rl-benchmarks repository](https://github.com/martius-lab/dynamic-rl-benchmarks). If you are interested in applying HiTS to a different task, then [this demo](https://github.com/nicoguertler/graph_rl/blob/master/demos/hits_drawbridge_env.py) in the Graph-RL repository is the best place to start. 

## Installation

We recommend using a [virtual environment](https://docs.python.org/3/tutorial/venv.html) with python3.7 or higher. Make sure pip is up to date. In the root directory of the repository execute:

```bash
pip install -r requirements.txt
```

## Usage

To render episodes with one of the pretrained policies execute in the root directory:

```bash
python -m scripts.run.render --algo hits --env Platforms

```
Available algorithms:
* hits
* hac
* sac

Available environments:
* AntFourRooms
* Drawbridge
* Pendulum
* Platforms
* Tennis2D
* UR5Reacher

A policy can be be trained from scratch by running:

```bash
python -m scripts.run.train --algo hits --env Platforms

```
To render episodes with a newly trained policy use:

```bash
python -m scripts.run.render --algo hits --env Platforms --newly_trained

```

To render an episode with the stochastic policy used during training:

```bash
python -m scripts.run.render --algo hits --env Platforms --newly_trained --stochastic

```

Hyperparameters and seeds can be found in the `graph_params.json` files in the `data` directory. The key `level_params_list` contains a list of the hyperparameters of all levels, starting with the lowest level.

## Results

How HiTS outperforms baselines in dynamic environments can be seen in [this video](https://youtu.be/JkPaI3uZU6c?t=287).

## How to cite

Please use the following BibTex entry.

```
@article{gurtler2021hierarchical,
  title={Hierarchical Reinforcement Learning with Timed Subgoals},
  author={G{\"u}rtler, Nico and B{\"u}chler, Dieter and Martius, Georg},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```
