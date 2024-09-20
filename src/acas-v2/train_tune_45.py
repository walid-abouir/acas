import os
import gymnasium as gym
import time
from stable_baselines3 import PPO
from acas_xu_45 import AcasEnv
from ray import tune
from ray.air import session
from stable_baselines3.common.vec_env import DummyVecEnv

# Define custom storage paths for models and logs
models_dir = "models/"
logdir = "logs/"

# Create directories if they don't exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Define the training function with custom paths
def train_ppo(config):
    # Create the environment
    env = AcasEnv(render_mode="human")
    env = DummyVecEnv([lambda: env]) 

    rewards_train=[]
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        ent_coef=config["ent_coef"],
        verbose=0,
        tensorboard_log=logdir  
    )

    # Train the model
    model.learn(total_timesteps=20000)

    # Evaluate the model to get the average reward
    num_episodes = 100
    total_reward = 0
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        rewards_train.append(total_reward)
    avg_reward = total_reward / num_episodes
    final_reward=rewards_train[-1]

    # Save the model in the custom models directory
    model_save_path = os.path.join(models_dir, f"ppo_model_{int(time.time())}")
    model.save(model_save_path)

    
    trial_dir = session.get_trial_dir()
    checkpoint_dir = os.path.join(trial_dir, "model")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save(os.path.join(checkpoint_dir, "ppo_model"))

    
    session.report({"reward_mean": avg_reward})



hyperparameter_search_space = {
    "learning_rate": tune.loguniform(1e-5, 1e-2),
    "n_steps": tune.choice([128, 256, 512]),
    "batch_size": tune.choice([64, 128, 256]),
    "gamma": tune.uniform(0.9, 0.9999),
    "ent_coef": tune.loguniform(1e-8, 0.1)
}

# Add the 'file://' scheme to the storage path
storage_path = f"file://{os.path.abspath(models_dir)}"


analysis = tune.run(
    train_ppo,
    config=hyperparameter_search_space,
    num_samples=1,  
    resources_per_trial={"cpu": 1, "gpu": 0},  
    storage_path=storage_path,  
    mode='max',
    metric=["reward_mean", "reward_episode"],
)

# Print the best hyperparameters found
print("Best hyperparameters found were: ", analysis.best_config)
