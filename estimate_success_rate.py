"""
Live simulation test for given number of episode and max episode length
Success rate is defined based on the goal_proximity_threshold
"""
from rlbench.environment import Environment
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import ReachTarget
from skimage.transform import resize
import numpy as np
import torch
import os

from models.cnn_fcn_64x64 import Net
model = Net()

downsample_images = True

cwd = os.getcwd()
model.load_state_dict(torch.load(os.path.join(cwd, 'cnn_fcn_64x64_2.pth'), map_location=torch.device('cpu')))
model.eval()
obs_config = ObservationConfig()
obs_config.set_all(True)

env = Environment(
    MoveArmThenGripper(
        arm_action_mode=JointVelocity(), gripper_action_mode=Discrete()),
    obs_config=obs_config,
    headless=False,
    static_positions = True)
env.launch()
task = env.get_task(ReachTarget)
dist_list = []
min_dist_list = []

num_episodes = 1
max_episode_steps = 100
goal_proximity_threshold = 0.1
num_successes = 0

for i in range(num_episodes): 
    _, obs = task.reset()
    min_dist = 1e8
    for j in range(max_episode_steps):
        obs1 = np.asarray(obs.left_shoulder_rgb)
        obs2 = np.asarray(obs.right_shoulder_rgb)
        obs3 = np.asarray(obs.overhead_rgb)
        obs4 = np.asarray(obs.wrist_rgb)
        obs5 = np.asarray(obs.front_rgb)
        imgs = [obs1, obs2, obs3, obs4, obs5]
        if downsample_images:
            for u in range(5):
                imgs[u] = resize(imgs[u], (64,64), preserve_range=True, anti_aliasing=True)
        input1 = np.stack(imgs)
        input1 = torch.Tensor(np.asarray([np.moveaxis(input1, 3, 1)]))
        input2 = torch.Tensor(np.asarray([np.asarray(obs.joint_positions)]))
        action = np.concatenate([model(x1=input1, x2=input2).detach().numpy()[0], [1.0]])
        gripper_position = obs.gripper_pose[:3]
        target_position =  obs.task_low_dim_state
        dist = np.linalg.norm(gripper_position - target_position)
        dist_list.append(dist)
        obs, reward, terminate = task.step(action)
        if dist < min_dist:
            min_dist = dist
        if dist < goal_proximity_threshold:
            num_successes += 1
            break
    min_dist_list.append(min_dist)

env.shutdown()
print("AVERAGE MIN DISTANCES: {0:0.2f}m".format(np.mean(np.array(min_dist_list))))
print(f"SUCCESS RATE after {num_episodes} episodes: {float(num_successes)/num_episodes*100}%")