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

"""Environment with the 90 degrees init"""
from acas_xu import AcasEnv
"""Environment with the 45 degrees init"""
#from acas_xu_45 import AcasEnv
"""Environment with the 180 degrees init"""
#from acas_xu_180 import AcasEnv
from acas_xu_continuous import AcasEnvContinuous
import gymnasium as gym



env = AcasEnv(render_mode="human")



#model for a 90 degrees initialization
model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/90_deg_agent/fixed_agent_model_2900000_steps.zip", env)

#model for a 45 degrees initialization
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/45_deg_agent/fixed_agent_model_3430000_steps.zip", env)

#model for a 180 degrees initialization
#model= PPO.load(os.path.dirname(os.path.realpath(__file__))+ "/models/180_deg_agent/fixed_agent_model_2900000_steps.zip", env)



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

mean_reward_dubins=-0.05
reward_mean = sum(mean_rewards)/len(mean_rewards)
reward_max = max(mean_rewards)
reward_agent_coc= -1.0

plt.figure(figsize=(6, 4))
plt.bar(['PPO', 'max PPO', 'COC agent', 'Dubins'], [reward_mean, reward_max, reward_agent_coc, mean_reward_dubins], color='skyblue', width=0.5)
plt.title('Moyenne des rewards pour plusieurs agents')
plt.grid(True, axis='y')
plt.xticks(rotation=45) 
plt.show()

nmacs_moy_dubins= 13
nmacs_moy_coc= 100

plt.figure(figsize=(6, 4))
plt.bar(['PPO', 'Dubins', 'COC agent'], [nmacs[-1], nmacs_moy_dubins, nmacs_moy_coc], color='skyblue', width=0.5)
plt.title('Pourcentage des NMACs')
plt.ylabel('NMACs (%)')
plt.grid(True, axis='y')
plt.xticks(rotation=45) 
plt.show()

print("""-----------------STATISTIQUES--------------------------------------""")
print('Moyenne des NMACS sur 100 épisodes :', nmacs[-1]/n_eval_episodes)

print('Moyenne des rewards :', sum(mean_rewards)/len(mean_rewards))
print('Maximum des rewards :', max(mean_rewards))
print('Minimum des rewards :', min(mean_rewards))