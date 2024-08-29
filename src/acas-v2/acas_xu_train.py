"""============================================================================================================="""
"""======================================================TRAINING==============================================="""
"""============================================================================================================="""

import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from acas_xu import AcasEnv




models_dir= f"models/PPO-{int(time.time())}"
logdir= f"logs/PPO-{int(time.time())}"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = AcasEnv()

term, trunc = False, False
env.reset()

model=PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)
TIMESTEPS= 2000
for i in range(1,100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS}")