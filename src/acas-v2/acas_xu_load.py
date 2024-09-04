import gymnasium
import numpy as np
from PIL import Image
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3 import A2C
from stable_baselines3 import SAC
from stable_baselines3 import DQN
import os
import pygame
import random
import torch
import time
import sys
from stable_baselines3.common.evaluation import evaluate_policy

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from acas_xu import AcasEnv
import gymnasium as gym




#models_dir= "models/PPO-1724842675"

#model_path = f"{models_dir}/100000.zip"


model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO_1/src_acas-v2_models_rl_model_16000_steps", env)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
vec_env = model.get_env()

obs= vec_env.reset()
term=False
trunc=False

while not (trunc or term):
    #env.render(render_mode="human")
    action, _= model.predict(obs)
    obs, reward, term, trunc, _ =env.step(action)
    vec_env.render(mode='human')

    