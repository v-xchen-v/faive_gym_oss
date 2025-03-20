from faive_gym.robot_hand import RobotHand
import torch

class XHandHoldDroppingBall(RobotHand):
    """
    A task where the robot hand learns to hold a ball and prevent it from dropping.
    The ball is dropped from a height, and the robot hand must catch it.
    """

    def check_termination(self):
        # override the function defined in RobotHand
        dist = torch.norm(self.object_pos - self.goal_pos, dim=-1)
        # envs where the cube was dropped
        self.dropped_buf = dist > self.cfg["rewards"]["fall_dist_threshold"]
        # check if exceeds the max episode length
        timeout_buf = self.progress_buf > self.max_episode_length - 1

        self.reset_goal_buf = self.dropped_buf | timeout_buf
        self.reset_buf = self.dropped_buf | timeout_buf
        
    def _reward_obj_linear_vel(self):
        """
        reward the movement of the object in x, y, z directions.
        """
        linear_vel = self.object_linvel_numerical
        
        # mean the abs of the linear velocity in x, y, z directions
        mean_linear_vel = torch.mean(torch.abs(linear_vel), dim=-1)
        # reward is the negative of the mean linear velocity
        reward = mean_linear_vel
        
        # reward to range [-0.1, 0]
        reward = torch.clamp(reward, 0, 1)
        
        dist = torch.norm(self.object_pos - self.goal_pos, dim=-1)
        # envs where the cube was dropped
        dropped_buf = dist > self.cfg["rewards"]["fall_dist_threshold"]
        
        # reward is 0 if the cube was dropped
        reward[dropped_buf] = 0
        return reward
        
    def _reward_obj_angular_vel(self):
        """
        reward the rotation of the object in x, y, z directions.
        """
        angular_vel = self.object_angvel_numerical
        
        # mean the abs of the angular velocity in x, y, z directions
        mean_angular_vel = torch.mean(torch.abs(angular_vel), dim=-1)
        
        # reward is the negative of the mean angular velocity
        reward = mean_angular_vel
        
        # reward to range [-0.1, 0]
        reward = torch.clamp(reward, 0, 1)
        
        dist = torch.norm(self.object_pos - self.goal_pos, dim=-1)
        # envs where the cube was dropped
        dropped_buf = dist > self.cfg["rewards"]["fall_dist_threshold"]
        
        # reward is 0 if the cube was dropped
        reward[dropped_buf] = 0
        return reward
        
    def _reward_obj_hand_dist(self):
        """
        reward the distance between the object and the hand.
        """
        # center is y-8 cm and y+5cm
        center_of_plam = self.hand_pose[:, :3].clone()
        center_of_plam[:, 1] -= 0.08
        center_of_plam[:, 2] += 0.05
        dist = torch.norm(self.object_pos - center_of_plam, dim=-1)
        
        # reward is the negative of the distance
        reward = dist
        
        # reward to range [-0.1, 0]
        # reward = torch.clamp(reward, 0, 1)
        
        max_dist = self.cfg["rewards"]["fall_dist_threshold"]
        reward = 1 - dist/max_dist
        # # envs where the cube was dropped
        # dropped_buf = dist > self.cfg["rewards"]["fall_dist_threshold"]
        
        # # reward is 0 if the cube was dropped
        # reward[dropped_buf] = 0
        return reward