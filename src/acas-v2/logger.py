from typing import List

from stable_baselines3.common.callbacks import BaseCallback


def callback_custom_metric_sb3(custom_metric_names: List[str]):

    class CustomMetricCallback(BaseCallback):

        def __init__(self, verbose=0):
            super(CustomMetricCallback, self).__init__(verbose)
            self.cumulated_metrics = {c: 0 for c in custom_metric_names}

        def _on_step(self) -> bool:

            for custom_metric_name in custom_metric_names:
                # Extract the custom metric from the info dictionary
                custom_metric = sum([infos.get(custom_metric_name, 0) for infos in self.locals['infos']])  # Handle several envs
                self.cumulated_metrics[custom_metric_name] += custom_metric
                self.logger.record(f"custom/{custom_metric_name}", self.cumulated_metrics[custom_metric_name])

            return True

    return CustomMetricCallback()

