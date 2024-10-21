from stable_baselines3 import PPO
from callback import TrainAndLoggingCallback

def train_agent(env, log_dir, checkpoint_dir, timesteps=1000):
    agent = PPO('CnnPolicy', env, tensorboard_log=log_dir, verbose=1,
                learning_rate=0.0001, n_steps=256)

    callback = TrainAndLoggingCallback(
        check_freq=1000, save_path=checkpoint_dir)

    agent.learn(total_timesteps=timesteps, callback=callback)
    return agent
