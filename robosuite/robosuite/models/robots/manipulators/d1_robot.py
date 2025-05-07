import numpy as np

from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.utils.mjcf_utils import xml_path_completion


class D1(ManipulatorModel):
    """
    D1 is a custom manipulator model for the D1 robot.

    This robot's MJCF XML (located at "robots/d1/robot.xml") has been modified to:
      - Remove the compiler tags.
      - Update all mesh file paths by prepending "meshes/".
      - Wrap the worldbody content inside a new <body name="base"> that includes a robotview camera.
      - Replace nested gripper bodies with a dummy "right_hand" body (with tunable pos and quat values).
      - Ensure all elements have a name attribute.

    Args:
        idn (int or str): A unique identification number or string for this robot instance.
    """

    arms = ["right"]

    def __init__(self, idn=0):
        # Loads the modified MJCF XML from "robots/d1/robot.xml"
        super().__init__(xml_path_completion("robots/d1/robot.xml"), idn=idn)

    @property
    def default_base(self):
        # The mount used in the modified XML
        return "RethinkMount"

    @property
    def default_gripper(self):
        # The default gripper to attach to the dummy right_hand body
        return {"right": "D1Gripper"}

    @property
    def default_controller_config(self):
        # The default controller configuration for the D1 robot
        return {"right": "default_d1"}

    @property
    def init_qpos(self):
        # Initial joint positions for a 6 DOF robot (adjust these if necessary)
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    @property
    def base_xpos_offset(self):
        # Offsets for placing the robot in different environments
        return {
            "bins": (-0.5, -0.1, 0),
            "empty": (-0.6, 0, 0),
            "table": lambda table_length: (-0.16 - table_length / 2, 0, 0),
        }

    @property
    def top_offset(self):
        # Top offset for visualization or attachment alignment
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        # Horizontal radius for collision or reach calculations
        return 0.5

    @property
    def arm_type(self):
        # Indicates a single-arm robot
        return "single"
