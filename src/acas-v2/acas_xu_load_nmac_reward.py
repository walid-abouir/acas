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
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from acas_xu_speeds import AcasEnvSpeeds
from acas_xu_heads import AcasEnvHeads
from acas_xu_45 import AcasEnv
from acas_xu_continuous import AcasEnvContinuous
import gymnasium as gym


# env = AcasEnvSpeeds(render_mode="human")
env = AcasEnv(render_mode="human")

#models_dir= "models/PPO-1724842675"

#model_path = f"{models_dir}/100000.zip"


#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO_1/src_acas-v2_models_rl_model_16000_steps", env)

#Head intruder randomized
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO-1725548382/fixed_agent_model_3000000_steps.zip", env)

# Speed ownship randomized
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/PPO-1725546805/3000000.zip", env)
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO-1725546805/fixed_agent_model_1500000_steps.zip", env)

# Speed Intruder randomized
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/PPO-1725546793/3000000.zip", env)

# Head ownship randomized
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/PPO-1725546779/3000000.zip", env)

# Interest time randomized
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/PPO-1725546779/3000000.zip", env)

# Two heads randomized 
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO-heads/fixed_agent_model_1700000_steps.zip", env)

"""SPEEDS angle 45"""
model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/logs/PPO-speeds-45/fixed_agent_model_2640000_steps.zip", env)



mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=1)
vec_env = model.get_env()



n_eval_episodes=100
nmac=0
nmacs= []
steps= []
mean_rewards=[]


for episode in range(n_eval_episodes):

    obs= vec_env.reset()
    term=False
    trunc=False
    episode_steps=0
    total_reward =0

    while not (trunc or term):
        #env.render(render_mode="human")
        action, _= model.predict(obs)
        obs, reward, term, trunc, _ =env.step(action)
        total_reward += reward
        episode_steps+=1

        
        

        if term == True :
            nmac+=1

        steps.append(len(steps))
        nmacs.append(nmac)
    #vec_env.render(mode='human')

    mean_rewards.append(total_reward)

print(len(steps))

plt.figure()
plt.plot(steps, nmacs)
plt.title("NMACs during timesteps")
plt.xlabel('steps')
plt.ylabel('NMACs')
plt.show()

plt.figure(figsize=(12, 6))
plt.bar(range(len(mean_rewards)), mean_rewards, color='skyblue')
plt.title('Moyenne des rewards par modèle')
plt.xlabel('Modèle (épisode)')
plt.ylabel('Reward moyenne')
plt.grid(True)
plt.show()

reward_mean = sum(mean_rewards)/len(mean_rewards)
reward_max = max(mean_rewards)
reward_agent_coc= -200

plt.figure(figsize=(6, 4))
plt.bar(['PPO', 'max PPO', 'COC agent'], [reward_mean, reward_max, reward_agent_coc], color='skyblue', width=0.5)
plt.title('Moyenne des rewards pour l\'agent PPO')
plt.ylabel('Reward moyenne')
plt.grid(True, axis='y')
plt.xticks(rotation=45) 
plt.show()

print("""-----------------STATISTIQUES--------------------------------------""")
print('Moyenne des NMACS sur 100 épisodes :', nmacs[-1]/n_eval_episodes)

print('Moyenne des rewards :', sum(mean_rewards)/len(mean_rewards))
print('Maximum des rewards :', max(mean_rewards))
print('Minimum des rewards :', min(mean_rewards))