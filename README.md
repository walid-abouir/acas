# Acas-XU

This environment is a custom environment designed to simulate a scenario where
two airplanes (ownship and intruder) are flying in a 2D plane. The goal is to 
avoid the collision between the airplanes by adjusting the heading of the 
ownship based on the relative position, speed, and heading of the intruder.

The environment is compatible with OpenAI Gym and can be used for reinforcement 
learning tasks. It includes methods for resetting the environment, taking a 
step in the environment, rendering the current state, and closing the environment.

<p align="center">
  <img src=media/simulation.gif width="75%"/>
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
Its value goes form - &pi; and &pi;
- The fifth value is theta, which is the relative heads between the two airplanes.
Its value goes form - &pi; and &pi;
- The sixth value is the last action taken by the agent, called "last_a". 
In the discrete version of the environment, it is an integer between 0 and 4.
In the continuous version of the environment, it's a float number, which is
added to ownship's head, to modify it, and avoid the collision.

## Action space

### Discrete version of the environment

The space of actions in the discrete version of the environment of one integer value
from 0 to 4. It is correlated to the angle added to the ownship's head to modify
its head :
- 0 &rarr; Clear Of Conflict (COC) : Maintain current heading
- 1 &rarr; Weak Left (WL) : Turn left (small angle)
- 2 &rarr; Weak Right (WR) : Turn right (small angle)
- 3 &rarr; Strong Left (SL) : Turn left (large angle)
- 4 &rarr; Strong Right (SR) : Turn right (large angle)

### Continuous version of the environment

The space of actions made available to the RL agent consists of one value from -3
to 3 degrees. It is correlated to the angle added to the ownship's head to modify
its head. The value -3 degrees means that the airplane turns strongly to the left,
and value 3 degrees means turning strongly  to the right.

## Initial episode conditions

At the start of each episode, initial positions and velocities for two airplanes
are set up. It first defines a fixed time of interest, when the collision will 
happen.


## Ending episode conditions