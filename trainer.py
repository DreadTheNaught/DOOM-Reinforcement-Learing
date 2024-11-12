from stable_baselines3 import PPO
from callback import TrainAndLoggingCallback
import json

with open('config.json', 'r') as config_file:
    config = json.load(config_file)


def train_agent(env, n_steps=2048):
    agent = PPO('CnnPolicy', env, tensorboard_log=config['log_dir'], verbose=1,
                learning_rate=0.0001, n_steps=n_steps)

    callback = TrainAndLoggingCallback(
        check_freq=1000, save_path=config["checkpoint_dir"])

    agent.learn(total_timesteps=config["training_timesteps"], callback=callback, progress_bar=True)
    return agent
