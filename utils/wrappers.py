import gymnasium as gym
import numpy as np
import cv2

IMAGE_SHAPE = (120, 160)
FRAME_SKIP = 1
REWARD_SCALING = 1
N_ENVS = 1

class ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE):
        super().__init__(env)
        self.image_shape = shape
        self.image_shape_reverse = shape[::-1]
        self.env.frame_skip = FRAME_SKIP

        # Create new observation space with the new shape
        print(env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (shape[0], shape[1], num_channels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation["screen"], self.image_shape_reverse)
        return observation


def wrap_env(env):
    env = ObservationWrapper(env)
    env = gym.wrappers.TransformReward(env, lambda r: r * REWARD_SCALING)
    return env
