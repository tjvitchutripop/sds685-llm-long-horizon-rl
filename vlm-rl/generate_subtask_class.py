import requests
import json
import xml.etree.ElementTree as ET
import base64
import numpy as np

TASK="MoveCubeToTable2"
task_file = f"/home/tj/Documents/robosuite/robosuite/environments/manipulation/{TASK}.py"

# Import decomposed tasks from XML file
xml_file = "decomp_tasks/"+TASK+".xml"
tree = ET.parse(xml_file)
subtasks = tree.getroot()

# Print the XML structure
for idx, subtask in enumerate(subtasks):
    subtask_name = subtask.attrib['name']
    if idx == 0:
        init_task_class = f"""
class {subtask_name}({TASK}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        # Append the init_task_class to the task_file
        with open(task_file, "a") as f:
            f.write(init_task_class)
    else:
        task_class = f"""  
class {subtask_name}({TASK}):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/{subtasks[idx-1].attrib['name']}.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
        """
        # Append the init_task_class to the task_file
        with open(task_file, "a") as f:
            f.write(task_class)

