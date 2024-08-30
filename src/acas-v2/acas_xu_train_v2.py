import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import time
import os
import gymnasium as gym
from acas_xu import AcasEnv
from logger import RewardLoggerCallback  # Assurez-vous d'importer le bon module


if __name__ == "__main__":
    # Définir le répertoire pour les modèles et les logs
    models_dir = f"/d/wabouir/acas-v2/src/acas-v2/models/PPO-{int(time.time())}"
    logdir = f"/d/wabouir/acas-v2/src/acas-v2/logs/PPO-{int(time.time())}"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Créer les environnements
    num_envs = 4
    env = SubprocVecEnv([lambda: AcasEnv() for _ in range(num_envs)])

    # Créer le modèle PPO
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    # Créer le callback
    callback = RewardLoggerCallback()

    # Entraîner le modèle
    TIMESTEPS = 1000000
    for i in range(1, 100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", callback=callback)
        model.save(f"{models_dir}/{TIMESTEPS*i}")

        obs = env.reset()
        total_rewards = [0] * num_envs
        done = [False] * num_envs

        while not all(done):
            action, _states = model.predict(obs)
            obs, rewards, terminated, truncated, info = env.step(action)
            total_rewards = [total_rewards[j] + rewards[j] for j in range(num_envs)]
            done = [d or t for d, t in zip(terminated, truncated)]

        print(f"Total rewards after {TIMESTEPS*i} timesteps: {sum(total_rewards)/num_envs}")
