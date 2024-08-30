"""============================================================================================================="""
"""======================================================TRAINING==============================================="""
"""============================================================================================================="""

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC
from stable_baselines3 import DQN

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from acas_xu import AcasEnv
import gymnasium as gym

def make_env():
    return AcasEnv()

if __name__ == "__main__":

    models_dir= f"models/PPO-{int(time.time())}"
    logdir= f"logs/PPO-{int(time.time())}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)



    num_envs = 4
    env = SubprocVecEnv([make_env for _ in range(num_envs)])

    term, trunc = False, False
    env.reset()

    model=PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
    TIMESTEPS= 1000000
    for i in range(1,100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")