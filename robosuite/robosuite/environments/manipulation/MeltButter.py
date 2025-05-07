from collections import OrderedDict

import numpy as np

from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import MultiTableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.observables import Observable, sensor
from robosuite.utils.placement_samplers import UniformRandomSampler, SequentialCompositeSampler
from robosuite.utils.transform_utils import convert_quat
from robosuite.environments.manipulation.libero_objects.articulated_objects import Microwave
from robosuite.environments.manipulation.libero_objects.google_scanned_objects import AkitaBlackBowl
from robosuite.environments.manipulation.libero_objects.hope_objects import Butter

class MeltButter(ManipulationEnv):
    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        initialization_noise="default",
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        lite_physics=True,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        camera_segmentations=None,  # {None, instance, class, element}
        renderer="mjviewer",
        renderer_config=None,
    ):
        # settings for table top
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = [(1, -1.5, 0.8), (1, 1.5, 0.8)]

        # reward configuration
        self.reward_scale = reward_scale
        self.reward_shaping = reward_shaping

        # whether to use ground-truth object states
        self.use_object_obs = use_object_obs

        # object placement initializer
        self.placement_initializer = placement_initializer

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types="default",
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
        )

    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # Adjust base pose accordingly according to the table offset
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        xpos = (xpos[0] + self.table_offset[0][0], xpos[1] + self.table_offset[0][1], xpos[2])
        # new_xpos = (xpos[0] - np.random.uniform(0.05, 0.75), xpos[1] + np.random.uniform(-0.5, 0.5), xpos[2])
        # self.robots[0].robot_model.set_base_xpos(new_xpos)
        self.robots[0].robot_model.set_base_xpos(xpos)

        # load model for table top workspace
        mujoco_arena = MultiTableArena(
            table_offsets= self.table_offset,
            table_full_sizes=[self.table_full_size,self.table_full_size],
            table_frictions=[self.table_friction,self.table_friction],
        )

        # Arena always gets set to zero origin
        mujoco_arena.set_origin([0, 0, 0])

        # initialize objects of interest
        self.microwave = Microwave()
        self.butter = BoxObject(
            name="butter",
            size = (0.015, 0.05, 0.015),
            rgba=[255/255, 255/255, 171/255, 1], 
            density=500,
        )
        self.bowl = AkitaBlackBowl()

        self.placement_initializer = SequentialCompositeSampler(name="ObjectSampler")
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.bowl.name}ObjectSampler",
                mujoco_objects=[self.bowl],
                # x_range=[-0.25, 0.25],
                # y_range=[-0.25, 0.25],
                x_range=[0.1, 0.1],
                y_range=[0.1, 0.1],
                rotation=None,
                ensure_object_boundary_in_range=True,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
            ))
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.butter.name}ObjectSampler",
                mujoco_objects=[self.butter],
                # x_range=[-0.25, 0.25],
                # y_range=[-0.25, 0.25],
                x_range=[-0.15, -0.15],
                y_range=[-0.15, -0.15],
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[0],
                z_offset=0.01,
            ))
        self.placement_initializer.append_sampler(
            sampler=UniformRandomSampler(
                name=f"{self.microwave.name}ObjectSampler",
                mujoco_objects=[self.microwave],
                # x_range=[-0.25, 0.25],
                # y_range=[-0.25, 0.25],
                x_range=[-0.03, 0.03],
                y_range=[-0.03, 0.03],
                rotation=50,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=self.table_offset[1],
                z_offset=0.01,
            ))

        # task includes arena, robot, and objects of interest
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=[self.microwave, self.bowl, self.butter],
        )

    def _setup_references(self):
        """
        Sets up references to important components. A reference is typically an
        index or a list of indices that point to the corresponding elements
        in a flatten array, which is how MuJoCo stores physical simulation data.
        """
        super()._setup_references()

        # Additional object references from this env
        self.bowl_body_id = self.sim.model.body_name2id(self.bowl.root_body)
        self.butter_body_id = self.sim.model.body_name2id(self.butter.root_body)
        self.microwave_body_id = self.sim.model.body_name2id(self.microwave.root_body)

    def _setup_observables(self):
        """
        Sets up observables to be used for this environment. Creates object-based observables if enabled

        Returns:
            OrderedDict: Dictionary mapping observable names to its corresponding Observable object
        """
        observables = super()._setup_observables()
        return observables


    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # Reset all object positions using initializer sampler if we're not directly loading from an xml
        if not self.deterministic_reset:

            # Sample from the placement initializer for all objects
            object_placements = self.placement_initializer.sample()

            # Loop through all objects and reset their positions
            for obj_pos, obj_quat, obj in object_placements.values():
                if "microwave" in obj.name:
                    self.sim.data.set_joint_qpos(obj.joints[1], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))
                else:
                    self.sim.data.set_joint_qpos(obj.joints[0], np.concatenate([np.array(obj_pos), np.array(obj_quat)]))

    def reward(self, action=None):
        return 0

    def visualize(self, vis_settings):
        """
        In addition to super call, visualize gripper site proportional to the distance to the cube.

        Args:
            vis_settings (dict): Visualization keywords mapped to T/F, determining whether that specific
                component should be visualized. Should have "grippers" keyword as well as any other relevant
                options specified.
        """
        # Run superclass method first
        super().visualize(vis_settings=vis_settings)

class pick_up_butter(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
          
class place_butter_in_bowl(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/pick_up_butter.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class pick_up_bowl_with_butter(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/place_butter_in_bowl.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class navigate_to_microwave(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/pick_up_bowl_with_butter.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class place_bowl_on_table(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/navigate_to_microwave.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class open_microwave_door(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/place_bowl_on_table.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class place_bowl_in_microwave(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/open_microwave_door.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
          
class close_microwave_door(MeltButter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_states = np.load("/home/tj/Documents/robosuite/robosuite/environments/manipulation/cached_states/place_bowl_in_microwave.npy")

    def reset(self):
        super().reset()
        self.sim.set_state_from_flattened(self.cached_states[np.random.randint(0, len(self.cached_states))])
        self.sim.forward() 

        return self._get_observations()
        