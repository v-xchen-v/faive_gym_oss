# This script is used to record the trajectory of the XHand DOF model
# The script will first play the model and then record the trajectory
# The recorded trajectory will be saved in a file

source ~/miniconda3/etc/profile.d/conda.sh
conda activate py38_gym_preview

cd /home/xichen/Documents/repos/faive_gym_oss/faive_gym

# define a checkpoint path variable
# let user input the checkpoint file name
read -p "Enter the full path of the checkpoint: " checkpoint_path
# set the user input as a variable
checkpoint_path=$checkpoint_path
# print the checkpoint path
echo "The checkpoint path is: $checkpoint_path"
# press enter to continue
read -p "Press enter to continue"
# run the model with the user input checkpoint
python train.py task=XHandP0 test=True num_envs=1 headless=False capture_video=True force_render=False checkpoint=$checkpoint_path

read -p "The record will be save at recording folder, check the video if satisfied, copy the path of npy for the record file name"
# let user input the record file name
read -p "Enter the full path of record file name: " record_file_name
# press enter to continue
read -p "Press enter to continue"
# set the user input as a variable
record_file_name=$record_file_name
python scripts/scale_and_animate_xhand_dof_traj.py --record_file_name $record_file_name --animate True