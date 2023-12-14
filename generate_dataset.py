import numpy as np

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
import pickle
from skimage.transform import resize
import datetime
import time

output_path = "dataset/downsampled/2/dataset"
num_demos = 50

for i in range(1):

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    env = Environment(
        action_mode=MoveArmThenGripper(
            arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
        obs_config=ObservationConfig(),
        headless=True)
    env.launch()

    task = env.get_task(ReachTarget)

    demos = task.get_demos(num_demos, live_demos=True)

    start_time = datetime.datetime.now().strftime("_%I-%M%p")
    dataset_main = {"rgbs": np.empty((0,5,64,64,3), np.uint8), "joint_positions": np.empty((0,7), np.float32) , "joint_velocities": np.empty((0,7), np.float32)}

    for episode in demos:
        for step in episode:

            obs1 = (resize(step.left_shoulder_rgb, (64,64), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
            obs2 = (resize(np.asarray(step.right_shoulder_rgb), (64,64), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
            obs3 = (resize(np.asarray(step.overhead_rgb), (64,64), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
            obs4 = (resize(np.asarray(step.wrist_rgb), (64,64), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
            obs5 = (resize(np.asarray(step.front_rgb), (64,64), preserve_range=True, anti_aliasing=True)).astype(np.uint8)
            joints = np.asarray([step.joint_positions], dtype=np.float32)
            obs = np.asarray([np.stack([obs1, obs2, obs3, obs4, obs5])], dtype=np.uint8)
            
            vel = np.asarray([step.joint_velocities], dtype=np.float32)

            dataset_main["rgbs"]= np.append(dataset_main["rgbs"], obs, 0)
            dataset_main["joint_positions"]= np.append(dataset_main["joint_positions"], joints, 0)
            dataset_main["joint_velocities"]= np.append(dataset_main["joint_velocities"], vel, 0)


    with open(output_path+start_time+".pkl", 'wb') as pickle_file:
        pickle.dump(dataset_main, pickle_file)

    env.shutdown()
    time.sleep(10)
