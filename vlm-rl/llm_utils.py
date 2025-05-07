import requests
import json
import os

def generate_init_reward_success_fn(composite_task_name, subtasks, current_subtask_idx):
    # Get list of subtask names
    subtask_names = []
    for idx, subtask in enumerate(subtasks):
        subtask_name = subtask.attrib['name']
        subtask_names.append(subtask_name)
    str_subtask_names = ", ".join(subtask_names)
    # Get prompt
    with open("prompts/init_reward_success_fn_gen.txt", "r") as file:
        reward_success_prompt = file.read()
    # Get utility function
    with open("prompts/utility_fns.txt", "r") as file:
        utility_functions = file.read()
    # Get composite task class
    with open("prompts/"+composite_task_name+"_class.txt", "r") as file:
        composite_task_class = file.read()
    # Send a request to OpenRouter API
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer YOUR_API_KEY",
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "anthropic/claude-3.7-sonnet",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": f"""
{reward_success_prompt}
Current Subtask: {subtask_names[current_subtask_idx]}
For context, the composite task is {composite_task_name} and all the subtasks in order are {str_subtask_names}.
{utility_functions}
{composite_task_class}
"""
            },
            ]
        }
        ],
        
    })
    )
    # Print the response JSON
    reward_success_code = response.json()["choices"][0]["message"]["content"]
    print("Successfully generated reward and success functions for subtask:", subtask_names[current_subtask_idx])
    # Save the reward_success_code to a file
    with open(f"reward_success_fns/{composite_task_name}_{subtask_names[current_subtask_idx]}_init.txt", "w") as f:
        f.write(reward_success_code)
    # Parse the code to get the reward and success functions
    reward_fn_code = reward_success_code[reward_success_code.find('def reward'):reward_success_code.find('def _check_success')-1]
    success_fn_code = reward_success_code[reward_success_code.find('def _check_success'):]
    return reward_fn_code, success_fn_code

def load_generated_reward_success_fn(composite_task_name, subtask_name):
    # Read the reward and success function code from the file
    with open(f"reward_success_fns/{composite_task_name}_{subtask_name}_init.txt", "r") as f:
    
        reward_success_code = f.read()
    # Parse the code to get the reward and success functions
    reward_fn_code = reward_success_code[reward_success_code.find('def reward'):reward_success_code.find('def _check_success')-1]
    success_fn_code = reward_success_code[reward_success_code.find('def _check_success'):]
    return reward_fn_code, success_fn_code