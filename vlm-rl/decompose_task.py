import requests
import json
import xml.etree.ElementTree as ET
import base64
import robosuite
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.manipulation import MeltButter, MoveCubeToTable2
import numpy as np
from PIL import Image

config = {
        "env_name": "MoveCubeToTable2",
        "robots": ["PandaOmron"],
        "controller_configs": load_composite_controller_config(controller="BASIC"),
    }


# Notice how the environment is wrapped by the wrapper
env = robosuite.make(
        **config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        camera_names=["birdview","sideview"],
        camera_heights=720,
        camera_widths=1080,
    )

observation = env.reset()
birdview_image = observation["birdview_image"]
sideview_image = observation["sideview_image"]
birdview_image = np.array(birdview_image)
sideview_image = np.array(sideview_image)
# flip the images
birdview_image = np.flip(birdview_image, axis=0)
sideview_image = np.flip(sideview_image, axis=0)
# Put them side by side
combined_image = np.concatenate((birdview_image, sideview_image), axis=1)
# Convert to PIL Image
combined_image = Image.fromarray(combined_image)
# Save the image
combined_image.save("images/"+config["env_name"]+".png")
# Close the environment
env.close()


# Convert scene image to base64
with open("images/"+config["env_name"]+".png", "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Get decomposition prompt from txt
with open("prompts/task_decomposition.txt", "r") as file:
    decomposition_prompt = file.read()
with open("prompts/"+config["env_name"]+"_desc.txt", "r") as file:
    task_description = file.read()

# Send a request to OpenRouter API
response = requests.post(
  url="https://openrouter.ai/api/v1/chat/completions",
  headers={
    "Authorization": "Bearer YOUR_API_KEY",
    "Content-Type": "application/json",
  },
  data=json.dumps({
    "model": "anthropic/claude-3.7-sonnet:thinking",
    "messages": [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": decomposition_prompt + task_description
          },
          {
            "type": "image_url",
            "image_url": {
              "url": "data:image/png;base64," + base64_image
            }
          }
        ]
      }
    ],
    
  })
)
print(response.json())  # Print the response JSON
bt_xml = response.json()["choices"][0]["message"]["content"]
print(bt_xml)

bt_xml = bt_xml[bt_xml.find('<'):]
bt_xml = bt_xml[:bt_xml.rfind('>')+1]

# Write the XML string to a file
with open("decomp_tasks/"+config["env_name"]+".xml", "w") as f:
    f.write(bt_xml)

print("XML file saved as", config["env_name"]+".xml")
