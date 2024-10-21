import gymnasium as gym
import numpy as np
import cv2

IMAGE_SHAPE = (60, 80)
FRAME_SKIP = 4
REWARD_SCALING = 0.01


class ObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = FRAME_SKIP

        # Adjust observation space with the new image shape
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8)

    def observation(self, observation):
        return cv2.resize(observation["screen"], self.image_shape_reverse)


def wrap_env(env, wrapper, reward_scaling):
    env = wrapper(env)
    env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env
