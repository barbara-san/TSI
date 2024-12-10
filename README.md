# TSI - 2024/2025

## Intelligent Collaborative Driving Lane Changing Decision Of Multi-intelligent Connected Vehicles - Paper Implementation and Related Experiments

### Overview

This project was developed as part of the *Topics in Intelligent Systems* curricular unit, during the 2024/2025 academic year, at the Faculty of Engineering of the University of Porto, by students António Cardoso, André Sousa and Bárbara Santos.

This project implements and evaluates deep reinforcement learning approaches for autonomous vehicle lane changing decisions in highway environments. The work replicates a research paper studying Intelligent Connected Vehicles (ICV) lane changing behavior, and investigates some additional experiments.

### Description

The project uses the `Highway-env` simulation environment to train and test deep reinforcement learning models for controlling autonomous vehicles. The main objectives areto:

- Develop models that enable vehicles to achieve safe speeds while avoiding collisions
- Compare different approaches including DQN (Deep Q-Network) and PPO (Proximal Policy Optimization)
- Evaluate performance across different traffic densities and numbers of agents
- Test different observation types (kinematics vs image-based)

### Usage

In order to use the provided code, please execute `pip install -r req.txt`.
