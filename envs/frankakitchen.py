import numpy as np
import cv2 
# from envs.wrappers.time_limit import TimeLimit
from absl import app
from PIL import Image
import gymnasium as gym
import gymnasium_robotics 

env_name = "FrankaKitchen-v1"  # 替换为你要检查的环境名称
if env_name not in gym.envs.registry.keys():
    gym.register_envs(gymnasium_robotics)


class FrankaKitchenEnv:
    metadata = {}

    def __init__(self, name, action_repeat=1, size=(64, 64), seed=0, tasks=['microwave' ]):
 
        self._env = gym.make('FrankaKitchen-v1', tasks_to_complete=tasks, render_mode='rgb_array')
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
        spaces['observation'] = gym.spaces.Box(-np.inf, np.inf, (59,), dtype=np.float64)
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
        obs["image"] = self.get_img()
        obs['observation'] = self.origin_obs['observation']
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = True
        done =  terminated or truncated
        info = {"discount": np.array(1, np.float32)} # TODO debug here  maybe the discount should be 0.95
        return obs, reward, done, info
    

    def get_img(self):
        image= self._env.render().copy()
        return cv2.resize(image, self._size)

    def reset(self):
        self.origin_obs, reset_info = self._env.reset()
        obs = {}
        # obs = dict(time_step.observation)
        # obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}

        obs['observation'] = self.origin_obs['observation']
        obs["image"] = self.get_img()
        obs["is_terminal"] = False
        obs["is_first"] = True

        return obs

    def render(self, *args, **kwargs):
        if kwargs.get("mode", "rgb_array") != "rgb_array":
            raise ValueError("Only render mode 'rgb_array' is supported.")
 




class FKWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.observation_space = self.env.observation_space
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape,
                        self.env.action_space.low.min()),
            high=np.full(self.env.action_space.shape,
                         self.env.action_space.high.max()),
            dtype=self.env.action_space.dtype,
        )

    def get_img(self):
        return self.env.render().copy()

    def reset(self):
        self.env.reset()
        rgb = self.get_img()  
        assert np.min(rgb) >= 0 
        return rgb

    def step(self, action):
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        # obs['rgb'] = rgb
        obs = self.get_img()

        return (obs, reward, terminated or truncated, info)
    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, args, **kwargs):
        return self.env.render()


def make_env(cfg):
    if 'fk-' not in cfg.task: # franka kitchen
        raise ValueError('no such task in language table ')


    assert cfg.get('obs', 'state') == 'rgb', 'FrankaKitchen only support for rgb obs '

    # print('init the task', cfg.task)
    env = gym.make('FrankaKitchen-v1', tasks_to_complete=['microwave' ], render_mode='rgb_array')
    env = FKWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=cfg.episode_length)
    env.max_episode_steps = env._max_episode_steps
    return env


if __name__ == '__main__':
    print('running the code ')
    env = make_env()
    env.reset()
    res = env.render()
    print(env.action_space)
    action = np.random.rand(9)
    print(action)

    obs, reward, terminated, truncated, info = env.step(action)
    print(obs)
    img = Image.fromarray(res) 
 
