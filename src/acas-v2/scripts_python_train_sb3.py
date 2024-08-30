import argparse
import time
import os
import random
import socket

import torch
import numpy as np
from stable_baselines3 import PPO, SAC, A2C, TD3, DDPG, DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv

import phd.env
from phd.logger import callback_custom_metric_sb3


root_path_logs = "/stck/troux/phd/runs/" if "spiro" in socket.gethostname() else "/home/troux/phd/runs/"

# Argument parser
parser = argparse.ArgumentParser(description='RL algorithm parameters')
parser.add_argument('--algorithm', type=str, choices=['PPO', 'SAC', 'A2C', 'TD3', 'DDPG', 'DQN'], default='PPO', help='RL algorithm to use')
parser.add_argument('--gym_id', type=str, default="PyFlyt/QuadX-Hover-v2", help='Environment ID')
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to use in parallel")
parser.add_argument('--verbose', type=int, default=0, help='Verbose level')
parser.add_argument('--seed', type=int, default=None, help='Random seed')
parser.add_argument('--total_timesteps', type=int, default=2_000_000, help='Total timesteps')
parser.add_argument('--log_metrics', type=str, nargs='+', help='The names of the custom metrics to be logged in Tensorboard (they have to be in infos)')
parser.add_argument('--reward_crash', type=int, help='The reward given when the drone crashes')


args = parser.parse_args()

if args.seed is None:
    args.seed = random.randint(1, 1000000)

# Ensure repoducibility
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def make_env():
    env = gym.make(args.gym_id, reward_crash=args.reward_crash)
    if args.gym_id in ["PyFlyt/QuadX-Waypoints-v2", "PyFlyt/QuadX-Pole-Waypoints-v2"]:
        env = FlattenWaypointEnv(env, context_length=2)
    return env

# Environment
# env = make_vec_env(make_env, n_envs=args.num_envs, seed=args.seed)
env = gym.make(args.gym_id)

algo = {'PPO': PPO, 'SAC': SAC, 'A2C': A2C, 'TD3': TD3, 'DDPG': DDPG, 'DQN': DQN}[args.algorithm]

# Select and initialize the RL algorithm
model = algo(
    policy='MlpPolicy',
    env=env,
    tensorboard_log=root_path_logs + args.gym_id,
    seed=args.seed,
    verbose=args.verbose,
)

def generate_run_name(args):
    run_name_parts = [args.algorithm, str(args.reward_crash), "SB3"]
    run_name_parts.append(f"{int(time.time())}")
    return "_".join(run_name_parts)

# Learning
run_name = generate_run_name(args)
if args.log_metrics is not None:
    callback = callback_custom_metric_sb3(args.log_metrics)
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, tb_log_name=run_name, callback=callback)
else:
    model.learn(total_timesteps=args.total_timesteps, progress_bar=False, tb_log_name=run_name)
model.save(os.path.join(root_path_logs, args.gym_id, run_name + "_1","agent"))
