'''
Scale the recorded xhand DoFs dof angles from [-1, 1] to [lower_limits, upper_limits],
then
1. Animate with urdfpy to check visually
2. Dump to .npy for further use.

Usage:
python faive_gym/scripts/scale_and_animate_xhand_dof_traj.py --record_file_name faive_gym/videos/selected_xhand_rollball_2025-02-20/XHand_2025-02-20_17-52-29/2025-02-20_17-52-34_dof_poses
'''

from urdfpy import URDF
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--record_file_name', type=str, default=
                    'faive_gym/videos/XHandHoldDroppingBall_2025-03-20_10-43-35/2025-03-20_10-43-38_dof_poses')
# add a bool argument to specify whether to animate or not
parser.add_argument('--animate', type=bool, default=True)
argparse = parser.parse_args()
args = parser.parse_args()

# file name
RECORD_FILE_NAME = argparse.record_file_name
if RECORD_FILE_NAME[-4:] == '.npy':
    RECORD_FILE_NAME = RECORD_FILE_NAME[:-4]
# data shape as: [num_envs, n_steps, n_dofs]
data = np.load(f'{RECORD_FILE_NAME}.npy')
(num_envs, n_steps, n_dofs) = data.shape
# remove the last n rows where data all 0
data = data[~np.all(data == 0, axis=-1)].reshape(num_envs, -1, n_dofs)

robot = URDF.load('./assets/urdf/xhand/xhand_right.urdf')

# for link in robot.links:
#     print(link.name)


for joint in robot.actuated_joints:
    print(joint.name)


indexs = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
]

names = [
    "right_hand_index_bend_joint",
    "right_hand_index_joint1",
    "right_hand_index_joint2",
    "right_hand_mid_joint1",
    "right_hand_mid_joint2",
    "right_hand_pinky_joint1",
    "right_hand_pinky_joint2",
    "right_hand_ring_joint1",
    "right_hand_ring_joint2",
    "right_hand_thumb_bend_joint",
    "right_hand_thumb_rota_joint1",
    "right_hand_thumb_rota_joint2",
]


def map_values(arr, new_min, new_max, old_min=-1, old_max=1):
    return new_min + (arr - old_min) * (new_max - new_min) / (old_max - old_min)

# # Example: Mapping values from range [10, 20] to [0, 1]
# arr = np.array([10, 12, 15, 18, 20])
# mapped_arr = map_values(arr, 10, 20, 0, 1)

# print(mapped_arr)


#
print(np.clip(data[0, :, indexs[2]], 0, 10))
robot.animate(cfg_trajectory={
    names[0]:  map_values(data[0, :, indexs[0]], -0.175, 0.175),
    names[1]:  map_values(data[0, :, indexs[1]], 0, 1.92),
    names[2]:  map_values(data[0, :, indexs[2]], 0, 1.92),
    names[3]:  map_values(data[0, :, indexs[3]], 0, 1.92),
    names[4]:  map_values(data[0, :, indexs[4]], 0, 1.92),
    names[5]:  map_values(data[0, :, indexs[5]], 0, 1.92),
    names[6]:  map_values(data[0, :, indexs[6]], 0, 1.92),
    names[7]:  map_values(data[0, :, indexs[7]], 0, 1.92),
    names[8]:  map_values(data[0, :, indexs[8]], 0, 1.92),
    names[9]:  map_values(data[0, :, indexs[9]], 0, 1.83),
    names[10]: map_values(data[0, :, indexs[10]], -1.05, 1.57),
    names[11]: map_values(data[0, :, indexs[11]], -0.175, 1.83),
})

stacked_arr = np.stack([
    map_values(data[0, :, indexs[0]], -0.175, 0.175),
    map_values(data[0, :, indexs[1]], 0, 1.92),
    map_values(data[0, :, indexs[2]], 0, 1.92),
    map_values(data[0, :, indexs[3]], 0, 1.92),
    map_values(data[0, :, indexs[4]], 0, 1.92),
    map_values(data[0, :, indexs[5]], 0, 1.92),
    map_values(data[0, :, indexs[6]], 0, 1.92),
    map_values(data[0, :, indexs[7]], 0, 1.92),
    map_values(data[0, :, indexs[8]], 0, 1.92),
    map_values(data[0, :, indexs[9]], 0, 1.83),
    map_values(data[0, :, indexs[10]], -1.05, 1.57),
    map_values(data[0, :, indexs[11]], -0.175, 1.83)], axis=1)
expanded_arr = np.expand_dims(stacked_arr, axis=0)
print(expanded_arr.shape)
np.save(f'{RECORD_FILE_NAME}_scale.npy', expanded_arr)
