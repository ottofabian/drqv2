# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import collections
import copy
from collections import deque
from typing import Any, NamedTuple, Tuple, Union
from gym import spaces
import dm_env
import gym
import mbrl_envs
import numpy as np
from dm_control import manipulation, suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs
from dm_env._environment import TimeStep
from mbrl_envs import ObsTypes
from mbrl_envs.dmc.dmc_mbrl_env import DMCMBRLEnv


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        if isinstance(attr, str):
            return getattr(self, attr)
        else:
            return tuple.__getitem__(self, attr)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]

        self._obs_spec = copy.deepcopy(env.observation_spec())
        self._obs_spec[pixels_key] = specs.BoundedArray(shape=np.concatenate(
            [[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name=pixels_key)

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        pixels = np.concatenate(list(self._frames), axis=0)
        time_step.observation[self._pixels_key] = pixels
        return time_step

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(wrapped_action_spec.shape,
                                               dtype,
                                               wrapped_action_spec.minimum,
                                               wrapped_action_spec.maximum,
                                               'action')

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(observation=time_step.observation,
                                step_type=time_step.step_type,
                                action=action,
                                reward=time_step.reward or 0.0,
                                discount=time_step.discount or 1.0)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def make(name, frame_stack, action_repeat, seed, obs_type: ObsTypes, pixels_key):
    domain, task = name.split('_', 1)
    # overwrite cup to ball_in_cup
    domain = dict(cup='ball_in_cup').get(domain, domain)
    # make sure reward is not visualized
    # if (domain, task) in suite.ALL_TASKS:
    #     # env = suite.load(domain,
    #     #                  task,
    #     #                  task_kwargs={'random': seed},
    #     #                  visualize_reward=False)
    #     pixels_key = 'image'
    # else:
    #     # name = f'{domain}_{task}_vision'
    #     # env = manipulation.load(name, seed=seed)
    #     pixels_key = 'front_close'

    env = mbrl_envs.make(domain, task, seed, obs_type, action_repeat=action_repeat, img_size=(84, 84))
    env = Gym2DM_Control(env, obs_type)
    # add wrappers
    env = ActionDTypeWrapper(env, np.float32)
    # TODO Propreception dtype wrapper
    # env = ActionRepeatWrapper(env, action_repeat)
    # env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    # add renderings for clasical tasks
    # if (domain, task) in suite.ALL_TASKS:
    #     # zoom in camera for quadruped
    #     camera_id = dict(quadruped=2).get(domain, 0)
    #     render_kwargs = dict(height=84, width=84, camera_id=camera_id)
    #     env = pixels.Wrapper(env,
    #                          pixels_only=True,
    #                          render_kwargs=render_kwargs)
    # stack several frames
    env = FrameStackWrapper(env, frame_stack, pixels_key)
    env = ExtendedTimeStepWrapper(env)
    return env


def box_to_spec(box: spaces.Box, **kwargs):
    low = box.low
    high = box.high
    dtype = box.dtype
    shape = box.shape
    return specs.BoundedArray(shape, dtype, low, high, **kwargs)


class ObsDtypeWrapper(dm_env.Environment):
    def __init__(self, env: dm_env.Environment, dtype):
        self._env = env
        wrapped_observation_spec = env.observation_spec()
        self._observation_spec = specs.BoundedArray(wrapped_observation_spec.shape,
                                                    dtype,
                                                    wrapped_observation_spec.minimum,
                                                    wrapped_observation_spec.maximum,
                                                    'observation')

    def step(self, action):
        timestep = self._env.step(action)
        # TODO convert dtype
        return timestep

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        timestep = self._env.reset()
        # TODO convert to dtype
        return timestep

    def __getattr__(self, name):
        return getattr(self._env, name)


class Gym2DM_Control(dm_env.Environment):

    def __init__(self, env: DMCMBRLEnv, obs_type: ObsTypes):
        self._env = env
        # self._obs_type = obs_type
        self._obs_keys = obs_type.split('_')

        observation_specs = collections.OrderedDict()
        for k, box in zip(self._obs_keys, self._env.observation_space):
            observation_specs[k] = box_to_spec(box, name=k)
        self._observation_spec = observation_specs
        self._action_spec = box_to_spec(self._env.action_space, name='action')

    def reset(self) -> TimeStep:
        obs, _ = self._env.reset()
        observation = collections.OrderedDict()
        for k, o in zip(self._obs_keys, obs):
            observation[k] = o
        return TimeStep(StepType.FIRST, None, None, observation)

    def step(self, action) -> TimeStep:

        obs, reward, terminated, truncated, info = self._env.step(action)

        observation = collections.OrderedDict()
        for k, o in zip(self._obs_keys, obs):
            observation[k] = o

        step_type = StepType.MID
        discount = 1.0

        if terminated or truncated:
            step_type = StepType.LAST
            discount = 0.0

            if truncated:
                physics = self._env._base_env.physics
                discount = self._env._base_env._env._task.get_termination(physics) or 1.0

        return TimeStep(step_type, reward, discount, observation)

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec
