import xml.etree.ElementTree as ET
from utils import update_reward_function, update_success_function, make_env
from llm_utils import load_generated_reward_success_fn
from robosuite.controllers import load_composite_controller_config
from stable_baselines3 import SAC
from robosuite.environments.manipulation.MoveCubeToTable2 import MoveCubeToTable2, pick_up_cube



TASK="MoveCubeToTable2"
TRIALS = 10
env_config = {
    "env_name": "pick_up_cube",
    "has_renderer": False,
    "render_camera":"sideview",
    "robots": ["PandaOmron"],
    "controller_configs": load_composite_controller_config(controller="BASIC"),
    "horizon": 500,
    "reward_scale": 1.0,
    "reward_shaping": True,
    "ignore_done": False,
    "has_offscreen_renderer":False,
}

# (0) IMPORT DECOMPOSED TASKS FROM XML FILE
xml_file = "decomp_tasks/"+TASK+".xml"
tree = ET.parse(xml_file)
subtasks = tree.getroot()
print("TASK:", TASK)
print("Subtasks loaded from XML file")

# (1) INITIALIZE ENVIRONMENT
subtask_name = subtasks[0].attrib['name']
reward_code, success_code = load_generated_reward_success_fn(TASK, subtask_name)
success = 0
env = make_env(env_config, reward_code, success_code)

# ----- EVALUATION LOOP -----
for n in range(TRIALS):
    obs = env.reset()
    obs = obs[0]
    prev_successs = True
    # (2) RUN POLICY FOR EACH SUBTASK
    for idx, subtask in enumerate(subtasks):
        print("Subtask:", subtask.attrib['name'])
        if prev_successs:
            subtask_name = subtasks[idx].attrib['name']
            # (2.1) Load the reward and success functions
            reward_code, success_code = load_generated_reward_success_fn(TASK, subtask_name)
            update_success_function(env, success_code)
            update_reward_function(env, reward_code)
            # (2.2) Load the model
            print("run_id:", subtask.attrib['run_id'])
            model = SAC.load(f"models/{subtask.attrib['run_id']}/best_model.zip", env=env)
            # (2.3) Run model in the environment
            for i in range(env_config["horizon"]):
                print("Step:", i)
                action, _ = model.predict(obs,  deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if env.unwrapped._check_success():
                    # input("Press Enter to continue...")
                    print("Successfully completed subtask:", subtask_name)
                    break
                if terminated or truncated:
                    print("Failed to complete subtask:", subtask_name)
                    prev_successs = False
                    break
        else:
            break
        # (3) CHECK SUCCESS FOR ALL SUBTASKS
        if prev_successs and idx == len(subtasks) - 1:
            success += 1
            print("Successfully completed all subtasks")

env.close()
print(f"Success rate on {TASK}:", success / TRIALS)
        
        


