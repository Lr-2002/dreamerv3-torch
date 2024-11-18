import numpy as np
import cv2
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

import gymnasium as gym 


class SimplerEnv:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), seed=0):
        self._env = simpler_env.make('google_robot_pick_coke_can')
 
        self._action_repeat = action_repeat
        self._size = size
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {}
        # for key, value in self._env.observation_spec().items():
        #     if len(value.shape) == 0:
        #         shape = (1,)
        #     else:
        #         shape = value.shape
        #     spaces[key] = gym.spaces.Box(-np.inf, np.inf, shape, dtype=np.float32)
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):

        action_space = gym.spaces.Box(
            low=np.full(self._env.action_space.shape,
                        self._env.action_space.low.min()),
            high=np.full(self._env.action_space.shape,
                         self._env.action_space.high.max()),
            dtype=self._env.action_space.dtype,
        )

        return action_space
    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        obs = {}
        self.origin_obs, reward, terminated, truncated, info = self._env.step(action)
        # for _ in range(self._action_repeat):
        #     time_step = self._env.step(action)
        #     reward += time_step.reward or 0
        #     if time_step.last():
        #         break
        obs["image"] = self.render()
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = True
        done =  terminated or truncated
        info = {"discount": np.array(1, np.float32)}
        return obs, reward, done, info
    

    def get_img(self, obs):
        image =  get_image_from_maniskill2_obs_dict(self._env, obs)
        return cv2.resize(image, self._size)


    def reset(self):
        self.origin_obs, reset_info = self._env.reset()
        obs = {}
        # obs = dict(time_step.observation)
        # obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["image"] = self.render()
        obs["is_terminal"] = False
        obs["is_first"] = True

        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
        return self.get_img(self.origin_obs)
