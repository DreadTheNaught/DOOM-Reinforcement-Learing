import gymnasium as gym
from utils.wrappers import wrap_env
from trainer import train_agent

CHECKPOINT_DIR = 'checkpoint/'
LOG_DIR = 'logs/'
TRAINING_TIMESTEPS = 1000


def main():
    env = gym.make("VizdoomBasic-v0")
    env = wrap_env(env)

    train_agent(env, LOG_DIR, CHECKPOINT_DIR, TRAINING_TIMESTEPS)
    env.close()


if __name__ == "__main__":
    main()
