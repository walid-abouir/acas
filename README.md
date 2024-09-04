# Acas-XU

This environment is a custom environment designed to simulate a scenario where
two airplanes (ownship and intruder) are flying in a 2D plane. The goal is to 
avoid the collision between the airplanes by adjusting the heading of the 
ownship based on the relative position, speed, and heading of the intruder.

The environment is compatible with OpenAI Gym and can be used for reinforcement 
learning tasks. It includes methods for resetting the environment, taking a 
step in the environment, rendering the current state, and closing the environment.

<p align="center">
  <img src="2aiplanes.png"/>
</p>

## Installation and prerequisites

### Installation via source code from Git repository

If you want to make specific changes to the source code or extend it with your
own functionalities this method will suit you.

```
git clone http://gitlab-dtis.onera/wabouir/acas-v2.git
cd acas-v2
pip install -e .
```

### Environneent prerequisites

Environment to run needs Python3 with Gym, Pygame, Numpy, and StableBaselines3 libraries.

```
pip install stable-baselines3 numpy gymnasium gym pygame
```

# Environment details

## Observation space

The observation space consists of six values :
- The first value represents the minimum distance between the two airplanes.
We call it "rho". It goes from -1e4 and 1e4. The agents receives this value that 
indicates if it's close to a collision with the intruder.
- The second and third values of the observation are the ownship's and intruder's 
speed. It goes from 0 and 300 m/sec.
- The fourth value is theta, which is the relative angle between the two airplanes.
Its value goes form - \pi and $\pi$


## Action space

### Discrete version of the environment

### Continuous version of the environment

## Initial episode conditions

## Ending episode conditions