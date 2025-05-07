import xml.etree.ElementTree as ET
import numpy as np
import os
import torch
import wandb
import datetime
from wandb.integration.sb3 import WandbCallback
import multiprocessing as mp
from utils import StopTrainingOnSuccessThreshold, vlm_evaluation, update_reward_function, update_success_function, make_env, get_relevant_objects, test_reward_success_function
from llm_utils import generate_init_reward_success_fn, load_generated_reward_success_fn
import robosuite
from robosuite.environments.manipulation import MoveCubeToTable2, MeltButter
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback

TASK="MoveCubeToTable2"
FORCE_REGEN=False # Force the generation of new reward and success functions even if they already exist
training_config = {
    "composite_task": TASK,
    "total_timesteps": 10_000_000,
    "policy_type": "SAC",   
    "net_arch": [512,512,512],
    "activation_fn": torch.nn.ReLU,
    "learning_rate": 0.001,
    "batch_size": 128,
    "target_update_interval": 5,
    "gamma": 0.99,
    "ent_coef": "auto",
    "device": "cuda",
    "n_envs": 15,
    "seed":0,
}
env_config = {
    "has_renderer": False,
    "has_offscreen_renderer": False,
    "robots": ["PandaOmron"],
    "controller_configs": load_composite_controller_config(controller="BASIC"),
    "horizon": 500,
    "reward_scale": 1.0,
    "reward_shaping": True,
    "ignore_done": False,
}

if __name__ == '__main__':
    mp.set_start_method('forkserver', force=True)
    experiment_name = TASK + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # (0) IMPORT DECOMPOSED TASKS FROM XML FILE
    xml_file = "decomp_tasks/"+TASK+".xml"
    tree = ET.parse(xml_file)
    subtasks = tree.getroot()
    subtask_run = {}
    print("TASK:", TASK)
    print("Subtasks loaded from XML file")
    failed_subtask = False

    # ----- TRAINING LOOP FOR EACH SUBTASK -----
    for idx, subtask in enumerate(subtasks):
        if failed_subtask:
            break
        subtask_name = subtasks[idx].attrib['name']
        env_config["env_name"] = subtask_name
        print("Subtask name:", subtask_name)

        # (1) GENERATE/LOAD INITIAL REWARD AND SUCCESS FUNCTIONS
        if not FORCE_REGEN and os.path.exists(f"reward_success_fns/{TASK}_{subtask_name}_init.txt"):
            print("Loading generated reward and success functions for subtask:", subtask_name)
            reward_code, success_code = load_generated_reward_success_fn(TASK, subtask_name)
        else:
            print("Generating reward and success functions for subtask:", subtask_name)
            reward_code, success_code = generate_init_reward_success_fn(TASK, subtasks, idx)
        # (1.1) Test the reward and success functions
        test_reward_success_function(env_config, reward_code, success_code)

        # (2) TRAIN THE MODEL ON THE SUBTASK
        # (2.1) Initialize wandb
        training_config["subtask"] = subtask_name
        run = wandb.init(
            project="vlm-bt-rl",
            group=experiment_name,
            config=training_config,
            sync_tensorboard=True,
            save_code=True,
        )
        # (2.2) Setup the environment
        vec_env = SubprocVecEnv([lambda: make_env(env_config, reward_code, success_code) for _ in range(training_config["n_envs"])])
        print("Environment setup complete for subtask:", subtask_name)
        # (2.3) Setup the model
        model = SAC( 
            "MlpPolicy",
            vec_env, 
            learning_rate=training_config["learning_rate"],
            batch_size=training_config["batch_size"],
            device=training_config["device"],
            gamma=training_config["gamma"],
            ent_coef=training_config["ent_coef"],
            policy_kwargs=dict(
                net_arch=training_config["net_arch"],
                activation_fn=training_config["activation_fn"],
            ),
            tensorboard_log=f"runs/{run.id}",
            seed = training_config["seed"],
            verbose=1,
        )
        print("Model setup complete for subtask:", subtask_name)
        print("Starting training for subtask:", subtask_name)
        model.learn(
            total_timesteps=training_config["total_timesteps"], 
            log_interval=1,
            progress_bar=True,
            callback=[
                WandbCallback(
                model_save_path=f"models/{run.id}",
                model_save_freq=100,
                verbose=2,
                ), 
                EvalCallback(
                    eval_env=vec_env,
                    best_model_save_path=f"models/{run.id}",
                    eval_freq=5000,
                    deterministic=True,
                    render=False,
                ),
                StopTrainingOnSuccessThreshold(
                    success_threshold=0.8,
                    eval_env=vec_env,
                    check_freq=5000,
                    n_eval_episodes=5,
                    verbose=0
                )],
        )
        print("Training complete for subtask:", subtask_name)
        run.finish()
        # (3) EVALUATE MODEL ON SUBTASK
        # (3.1) Setup the environment
        env = make_env(env_config, reward_code, success_code)
        state_size = env.unwrapped.sim.get_state().flatten().shape
        # (3.2) Load the model
        model = SAC.load(f"models/{run.id}/best_model.zip", env=env)
        # (3.3) Test the model
        success = 0
        obs = env.reset()
        for i in range(10):
            obs = env.reset()
            obs = obs[0]
            for i in range(env_config["horizon"]):
                action, _ = model.predict(obs)
                obs, reward, done, info = env.step(action)
                if reward == 1:
                    success += 1
                    break
                if done:
                    obs = env.reset()
        success_rate = success / 10
        print(f"Success rate on {subtask_name}:", success_rate)
        if success_rate >= 0.8:
            print(f"Success rate on {subtask_name} is above 80%. Collecting cached states.")
            # (4) COLLECT CACHED STATES
            states = np.zeros((1000, state_size))
            n_success = 0
            while n_success < 1000:
                obs = env.reset()
                obs = obs[0]
                for i in range(500):
                    action, _ = model.predict(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    if env.unwrapped._check_success():
                        cached_state = env.unwrapped.sim.get_state().flatten()    
                        states[n_success] = cached_state
                        n_success += 1
                        break
            np.save(f"/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/{TASK}/{subtask_name}.npy", states)
            print("Collected cached states for subtask:", subtask_name)
                
    print("Training complete for all subtasks")

    # (5) UPDATE XML FILE WITH RUN ID FOR EACH SUBTASK
    for subtask in subtasks:
        subtask_name = subtask.attrib['name']
        if subtask_name in subtask_run:
            subtask.attrib['run_id'] = subtask_run[subtask_name]
        else:
            subtask.attrib['run_id'] = "None"
    tree.write(xml_file)
    print("XML file updated with run IDs for each subtask")