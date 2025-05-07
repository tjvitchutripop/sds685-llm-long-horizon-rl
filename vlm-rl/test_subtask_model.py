import xml.etree.ElementTree as ET
import numpy as np
import os
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
import multiprocessing as mp
from utils import update_reward_function, update_success_function, make_env
from llm_utils import generate_init_reward_success_fn, load_generated_reward_success_fn
import robosuite
from robosuite.environments.manipulation import MoveCubeToTable2, MeltButter
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

TASK="MoveCubeToTable2"

env_config = {
    "has_renderer": True,
    "robots": ["PandaOmron"],
    "controller_configs": load_composite_controller_config(controller="BASIC"),
    "horizon": 500,
    "reward_scale": 1.0,
    "reward_shaping": True,
    "ignore_done": False,
    "has_offscreen_renderer":False,
    "render_camera": "table1",
}

xml_file = "decomp_tasks/"+TASK+".xml"
tree = ET.parse(xml_file)
subtasks = tree.getroot()
print("TASK:", TASK)
print("Subtasks loaded from XML file")

subtask_name = subtasks[0].attrib['name']
env_config["env_name"] = subtask_name
reward_code, success_code = load_generated_reward_success_fn(TASK, subtask_name)

env = make_env(env_config, reward_code, success_code)
print(env.observation_space.shape)
# (3.2) Load the model
model = SAC.load(f"models/oas2u16q/best_model.zip", env=env)
# (3.3) Test the model
success = 0
obs = env.reset()
print("Start testing the model")
for i in range(10):
    print("Test episode:", i)
    obs = env.reset()
    obs = obs[0]
    for i in range(env_config["horizon"]):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.unwrapped._check_success():
            success += 1
            print("Success!")
            input("Press Enter to continue...")
            break
        if terminated or truncated:
            print("Episode terminated or truncated")
            break
             
env.close()
success_rate = success / 10
print(f"Success rate on {subtask_name}:", success_rate)