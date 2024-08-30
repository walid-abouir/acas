from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.ep_rewards = []  # List to store rewards for each episode

    def _on_step(self) -> bool:
        # Collect rewards for each episode
        if 'rewards' in self.locals:
            episode_rewards = self.locals['rewards']
            self.ep_rewards.extend(episode_rewards)
        return True

    def _on_rollout_end(self) -> None:
        # Optional: log episode rewards at the end of each rollout
        pass

    def _on_training_end(self) -> None:
        # Log the mean reward of all episodes at the end of training
        if self.ep_rewards:
            mean_reward = np.mean(self.ep_rewards)
            self.logger.record('ep_rew_mean', mean_reward)
            self.logger.dump(self.num_timesteps)

