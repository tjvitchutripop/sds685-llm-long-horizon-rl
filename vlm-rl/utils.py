import types
import os
import json
import base64
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import torch
import datetime
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

def encode_image(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def update_reward_function(env, reward_code):
    # Create a local namespace
    namespace = {'np': np}
    # Execute the string as code
    exec(reward_code, globals(), namespace)
    # Get the plain function
    reward_func = namespace["reward"]
    # If this is a wrapped environment, update the unwrapped environment
    if hasattr(env, 'unwrapped'):
        # Bind it to the unwrapped environment instance
        bound_method = types.MethodType(reward_func, env.unwrapped)
        # Replace the environment's reward method
        env.unwrapped.reward = bound_method
    else:
        # Bind it to the environment instance
        bound_method = types.MethodType(reward_func, env)
        # Replace the environment's reward method
        env.reward = bound_method

def update_success_function(env, success_code):
    # Create a local namespace
    namespace = {'np': np}
    # Execute the string as code
    exec(success_code, globals(), namespace)
    # Get the plain function
    success_func = namespace["_check_success"]
    # If this is a wrapped environment, update the unwrapped environment
    if hasattr(env, 'unwrapped'):
        # Bind it to the unwrapped environment instance
        bound_method = types.MethodType(success_func, env.unwrapped)
        # Replace the environment's success method
        env.unwrapped._check_success = bound_method
    else:
        # Bind it to the environment instance
        bound_method = types.MethodType(success_func, env)
        # Replace the environment's success method
        env._check_success = bound_method

def test_reward_success_function(env_config, reward_code, success_code):
    env = make_env(env_config, reward_code, success_code)
    try:
        _ = env.reset()
        env.step(np.zeros(env.action_space.shape))
    except Exception as e:
        print("Error with reward and success function:", e)
    env.close()

def make_env(config, reward_code, success_code):
    env = robosuite.make(
        **config,
        control_freq=20,
        use_camera_obs=False,
    )   
    try:
        update_reward_function(env, reward_code)
        update_success_function(env, success_code)
    except Exception as e:
        print("Error with reward and success function:", e)
    env = GymWrapper(env)
    return env

def eval_model(env_config, model, reward_code, success_code, num_episodes=10):
    curr_success = 0
    env = make_env(env_config, reward_code, success_code)
    for i in range(num_episodes):
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("success", False):
            curr_success += 1
            break
        if terminated or truncated:
            break
    env.close()
    curr_success = curr_success / num_episodes
    print(f"Current success rate: {curr_success:.2f}")
    return curr_success


def remove_backticks(llm_response):
    """
    Remove all backtick characters from an LLM response string,
    including triple backtick code blocks.
    
    Args:
        llm_response (str): The string containing the LLM response
        
    Returns:
        str: The response with all backtick characters removed
    """
    # First remove all triple backtick code blocks (with or without language specifier)
    import re
    # Replace triple backticks with or without language specifier with empty string
    cleaned_text = re.sub(r'```[\w]*\n|```', '', llm_response)
    
    # Then remove any remaining single backticks
    cleaned_text = cleaned_text.replace("`", "")
    
    return cleaned_text


class StopTrainingOnSuccessThreshold(BaseCallback):
    def __init__(self, success_threshold=0.8, eval_env=None, check_freq=5000, n_eval_episodes=5, verbose=1):
        super().__init__(verbose)
        self.success_threshold = success_threshold
        self.eval_env = eval_env
        self.check_freq = check_freq
        self.n_eval_episodes = n_eval_episodes
        self.success_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        successes = []
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()  
            if isinstance(obs, tuple):
                obs = obs[0]
            dones_table = np.full(self.eval_env.num_envs, False)
            success_this_episode = False  
            while not all(dones_table):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.eval_env.step(action)
                for idx, info in enumerate(infos):
                    if info["success"]==True and dones_table[idx]==False:
                        successes.append(1)
                        dones_table[idx] = True   
                    if dones_table[idx]==False and dones[idx]:
                        successes.append(0)
                        dones_table[idx] = True

        avg_success = np.mean(successes)
        self.logger.record("eval/success_rate", avg_success)
        self.success_history.append(avg_success)

        if self.verbose:
            print(f"[Callback] Success rate over last {self.n_eval_episodes} episodes: {avg_success:.2f}")

        if avg_success >= self.success_threshold:
            print(f"[Callback] Success threshold of {self.success_threshold} reached. Stopping training.")
            return False  # Returning False stops training

        return True