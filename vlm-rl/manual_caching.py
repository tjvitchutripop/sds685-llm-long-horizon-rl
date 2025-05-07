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
    "robots": ["PandaOmron"],
    "controller_configs": load_composite_controller_config(controller="BASIC"),
    "horizon": 500,
    "reward_scale": 1.0,
    "reward_shaping": True,
    "ignore_done": False,
    "has_offscreen_renderer":False
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
state_size = env.unwrapped.sim.get_state().flatten().shape
# Load the model
model = SAC.load(f"models/oas2u16q/best_model.zip", env=env)
# CACHING SUCCESS STATES
states = np.zeros((100, state_size))
print("Start caching success states")
n_success = 0
while n_success < 100:
    obs = env.reset()
    obs = obs[0]
    for i in range(500):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if env.unwrapped._check_success():
            cached_state = env.unwrapped.sim.get_state().flatten()
            env.reset()
            env.sim.set_state_from_flattened(cached_state)
            env.sim.forward() 
            grasp_state = action[-1]
            action = action*0
            action[-1] = grasp_state
            for j in range(100):
                env.step(action)
            if env.unwrapped._check_success():         
                states[n_success] = cached_state
                n_success += 1
            break
np.save(f"/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/{subtask_name}.npy", states)
print("Collected cached states for subtask:", subtask_name)
env.close()