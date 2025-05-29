# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.envs import DirectRLEnv
from omni.isaac.lab.sensors import ContactSensor, RayCaster

from .unitree_go2_env_cfg import UnitreeGo2FlatEnvCfg, UnitreeGo2RoughEnvCfg
import matplotlib.pyplot as plt
import numpy as np
import torch

def euler_to_quaternion(roll, pitch, yaw):
        """
        Convert Euler angles (roll, pitch, yaw) to quaternions (x, y, z, w).
        """
        cr = torch.cos(roll / 2)
        sr = torch.sin(roll / 2)
        cp = torch.cos(pitch / 2)
        sp = torch.sin(pitch / 2)
        cy = torch.cos(yaw / 2)
        sy = torch.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return torch.stack((x, y, z, w), dim=-1) 


class UnitreeGo2Env(DirectRLEnv):
    cfg: UnitreeGo2FlatEnvCfg | UnitreeGo2RoughEnvCfg

    def __init__(self, cfg: UnitreeGo2FlatEnvCfg | UnitreeGo2RoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "controlled_landing",
                "impact_penalty",
                "stand_reward",
                "upright_reward",
                "feet_stability",
                "undesired_contact",
                "torso_height_penalty",
                "survival_reward",
                "balanced_contact",
                "belly_up_penalty",
                "foot_tapping_penalty",
                "continuous_contact_reward",
                "foot_lift_penalty",
                "symmetry_penalty",
                "full_contact_reward",
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._calf_ids, _ = self._contact_sensor.find_bodies(".*_calf")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies([".*_thigh","base","Head_upper","Head_lower"])
        self._undesired_contact_body_ids1, _ = self._contact_sensor.find_bodies([".*_thigh","Head_upper","Head_lower"])

        # ✅ Get leg joint indices for computing leg extension reward
        self._leg_joint_ids, _ = self._robot.find_joints([".*_thigh_joint", ".*_calf_joint"])

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, UnitreeGo2RoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        """Modify spawn state after 150 steps and debug step count."""

        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

        # # # Adjust episode length buffer based on the condition
        # # adjusted_episode_length = self.episode_length_buf[0] 
        # # if adjusted_episode_length < 0:
        # #     adjusted_episode_length += 999
        # # else:
        # #     adjusted_episode_length = adjusted_episode_length-643+998
        # # self.episode_length_buf[0] = adjusted_episode_length

        # # Print current step count and elapsed time
        # print(f"📌 Current Step: {self.episode_length_buf[0]}")
        # print(f"📌 Elapsed Simulation Time: {self.episode_length_buf[0] * self.step_dt:.2f}s")

        # # Update state only when 150 steps have been reached
        # if self.episode_length_buf[0] >= 150:
        #     robot_state = self._robot.data.root_state_w.clone()

        #     # Modify spawn height, velocity
        #     robot_state[:, 2] = 3.0  # New height
        #     robot_state[:, 7:10] = torch.tensor([2.0, 0.0, 0.0], device=self.device)  # New linear velocity
        #     robot_state[:, 10:13] = torch.tensor([0.0, 0.5, 0.0], device=self.device)  # New angular velocity

        #     # Apply new state
        #     self._robot.write_root_pose_to_sim(robot_state[:, :7])
        #     self._robot.write_root_velocity_to_sim(robot_state[:, 7:])

        #     print(f"🔹 Updated Spawn at Step {self.episode_length_buf[0]}:")
        #     print(f"   📌 Height: {robot_state[0, 2].item()}m")
        #     print(f"   📌 Linear Velocity: {robot_state[0, 7:10].tolist()}")
        #     print(f"   📌 Angular Velocity: {robot_state[0, 10:13].tolist()}")
 


    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        height_data = None
        if isinstance(self.cfg, UnitreeGo2RoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,
                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        """Reward function for stable landing and standing task with detailed logging."""

        # 🟢 Step 1: Encourage controlled descent (not free-fall)
        downward_vel = self._robot.data.root_lin_vel_b[:, 2]
        controlled_landing_reward = torch.exp(-torch.square(downward_vel + 0.5))

        # 🟢 Step 2: Penalize hard landings (high impact force)
        impact_force = torch.norm(self._contact_sensor.data.net_forces_w[:, self._feet_ids, :], dim=-1).max(dim=1)[0]
        impact_penalty = torch.clamp(impact_force - 50.0, min=0.0)

        # 🟢 Step 3: Reward balance - torso upright
        world_up = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        robot_up = -self._robot.data.projected_gravity_b / 9.81
        upright_alignment = torch.sum(world_up * robot_up, dim=-1)
        upright_reward = torch.where(
            upright_alignment > 0, torch.exp(upright_alignment - 1), torch.zeros_like(upright_alignment)
        )

        # 🟢 Step 4: Reward feet staying on the ground
        feet_contact = torch.sum(self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids], dim=1)
        # feet_stability_reward = torch.exp(-torch.abs(feet_contact - 4.0))
        feet_stability_reward = torch.where(feet_contact == 4, 1.0, -0.25)

        # ✅ Step 5: Check if the robot has landed (low height and velocity)
        landed = (self._robot.data.root_pos_w[:, 2] < 0.5) & (torch.abs(downward_vel) < 0.1)

        # ✅ Step 6: Belly-Up Penalty (After Landing)
        robot_orientation = self._robot.data.root_quat_w  # Quaternion [x, y, z, w]
        belly_up = (robot_orientation[:, 2] < -0.7) & landed  # Only penalize after landing
        belly_up_penalty = torch.where(belly_up, -50.0, 0.0)  # Heavy penalty for belly-up posture

        # ✅ Penalize non-feet body parts touching ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        non_feet_contact = torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        undesired_contact_penalty = 10 * torch.sum(non_feet_contact, dim=1)

        # 🟢 Step 7: Survival Reward (Only when 4 feet contact and no undesired contact)
        survival_reward = torch.where(
            (feet_contact == 4) & (torch.sum(non_feet_contact, dim=1) == 0),
            0.25 * feet_contact,
            0.0
        )

        # 🟢 Step 8: Penalize for torso height less than 0.5 meters
        min_torso_height = 0.2
        current_torso_height = self._robot.data.root_pos_w[:, 2]
        height_difference = current_torso_height - min_torso_height
        torso_height_penalty = torch.where(
            height_difference < 0, -torch.exp(-5 * height_difference) + 1, torch.zeros_like(height_difference)
        )

        # 📌 Aggregate rewards
        rewards = {
            "controlled_landing": controlled_landing_reward * self.cfg.z_velocity_scale * self.step_dt,
            "impact_penalty": impact_penalty * self.cfg.impact_penalty_scale * self.step_dt,
            "upright_reward": upright_reward * self.cfg.upright_reward_scale * self.step_dt,
            "feet_stability": feet_stability_reward * self.cfg.feet_stability_scale * self.step_dt,
            "undesired_contact": undesired_contact_penalty * self.cfg.undersired_contact_reward_scale * self.step_dt,
            "torso_height_penalty": -torso_height_penalty * self.cfg.torso_height_penalty_scale * self.step_dt,
            "survival_reward": survival_reward * self.cfg.survival_reward_scale * self.step_dt,
            "belly_up_penalty": belly_up_penalty * self.step_dt  # ✅ Heavy penalty after landing
        }

        # ✅ Compute final reward sum
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # ✅ Logging: update episode sums for each reward component
        for key, value in rewards.items():
            self._episode_sums[key] += value

        # 🔹 Print individual reward components for debugging
        print("\n🔹 Step Reward Components:")
        for key, val in rewards.items():
            print(f"  {key}: {val.mean().item():.4f}")

        # 🔹 Compute cumulative reward across the episode
        cumulative_reward = sum(val.mean().item() for val in self._episode_sums.values())
        print(f"✅ Cumulative Reward so far: {cumulative_reward:.4f}\n")

        return reward

    
    def _plot_reward_components(self):
        # Prepare data for plotting
        components = list(self._episode_sums.keys())
        values = [val.mean().item() for val in self._episode_sums.values()]

        # Create bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(components, values)
        plt.title('Reward Components for the Episode')
        plt.xlabel('Reward Components')
        plt.ylabel('Cumulative Reward')
        plt.xticks(rotation=45, ha='right')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()



    # def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    #     time_out = self.episode_length_buf >= self.max_episode_length - 1
    #     net_contact_forces = self._contact_sensor.data.net_forces_w_history
    #     died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
    #     return died, time_out

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Determine when an episode should terminate."""

        # ✅ Timeout condition
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # ✅ Get contact forces
        net_contact_forces = self._contact_sensor.data.net_forces_w_history

        # ✅ 1️⃣ Belly-Up Termination (Robot flipped)
        robot_orientation = self._robot.data.root_quat_w  # Quaternion [x, y, z, w]
        belly_up = robot_orientation[:, 2] < -0.7  # Z-axis inverted → belly up


        # ✅ 3️⃣ Undesired Body Contact (e.g., thighs)
        undesired_contacts = torch.sum(
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids1], dim=-1), dim=1)[0] > 1.0,
            dim=1
        )

        # ✅ 4️⃣ Termination Conditions
        flipped = belly_up | (undesired_contacts > 0)

        # ✅ Combine conditions
        died = flipped  # Episode ends if flipped or head hits the ground

        return died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            # Spread out resets to avoid spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # 🟢 1️⃣ Randomize robot height between 0 and 4 meters
        default_root_state = self._robot.data.default_root_state[env_ids]
        # random_heights = torch.rand(len(env_ids), device=self.device) * 4.0  # Height between 0 and 4 meters
        # default_root_state[:, 2] = random_heights  # Set Z-axis height

        # # 🟢 2️⃣ Randomize x velocity between 0 and 2 m/s
        # random_x_velocity = (torch.rand(len(env_ids), device=self.device) * 3.5) - 0.5  # Range [-0.5, 3.0]
        # default_root_state[:, 7] = random_x_velocity  # Set linear velocity in X direction

        # # 🟢 3️⃣ Randomize y (lateral) velocity between -1 and +1 m/s
        # random_y_velocity = (torch.rand(len(env_ids), device=self.device) * 1.0) - 1.0  # Velocity between -1 and +1 m/s
        # default_root_state[:, 8] = random_y_velocity  # Set linear velocity in Y direction

        # # 🟢 4️⃣ Calculate omega_max based on height (2 * pi * sqrt(5/h))
        # omega_max = torch.pi * np.sqrt(2.5) * random_heights  # Avoid division by zero

        # # 🟢 5️⃣ Randomize pitch angular velocity (omega_y) between -omega_max/3 and +omega_max/3
        # random_omega_y = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * omega_max / 3
        # default_root_state[:, 11] = random_omega_y  # Set angular velocity around Y-axis (pitch)

        # # 🟢 6️⃣ Randomize roll angular velocity (omega_x) between -omega_max/5 and +omega_max/5
        # random_omega_x = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * omega_max / 5
        # default_root_state[:, 10] = random_omega_x  # Set angular velocity around X-axis (roll)

        # # Import PyTorch for quaternion conversion

        # # 🟢 7️⃣ Randomize yaw orientation between -2π and 2π
        # random_yaw = (torch.rand(len(env_ids), device=self.device) * 4.0 * torch.pi) - (2.0 * torch.pi)

        # # # 🟢 8️⃣ Randomize pitch orientation between 4.9 * h + π/8
        # c = torch.where(torch.abs(random_heights) < 1)
        # random_pitch_max = (1.57*(random_heights - 1))
        # random_pitch_max[c] = 
        
        # a = torch.where(torch.abs(random_pitch_max) > 1.57)
        # random_pitch_max[a] = torch.sign(random_pitch_max[a])*1.57
        

        # random_pitch = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * random_pitch_max

        # # 🟢 9️⃣ Randomize roll orientation between 7.2 * h + π/8
        # random_roll_max = (0.5*(random_heights - 1))

        
        # b = torch.where(torch.abs(random_roll_max) > 1)
        # random_roll_max[b] = torch.sign(random_roll_max[b])*1

        # random_roll_max[c] = 0
        
        # random_roll = (torch.rand(len(env_ids), device=self.device) * 2.0 - 1.0) * random_roll_max

        # # 🟢 8️⃣ Randomize pitch orientation between -π/3 and π/3
        # # random_pitch = (torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi / 3)) - (torch.pi / 3)

        # # # 🟢 9️⃣ Randomize roll orientation between -π/3 and π/3
        # # random_roll = (torch.rand(len(env_ids), device=self.device) * (2.0 * torch.pi / 3)) - (torch.pi / 3)

        # # 🟢 🔟 Convert Euler angles to quaternions
        # random_orientations = euler_to_quaternion(random_roll, random_pitch, random_yaw)

        # # 🟢 1️⃣1️⃣ Apply randomized orientation
        # default_root_state[:, 3:7] = random_orientations

        # Update root state and apply the changes
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Update root state and joint positions
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)

        # Reset joint positions and velocities
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # # Sample new commands
        # self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        # # Reset robot state
        # joint_pos = self._robot.data.default_joint_pos[env_ids]
        # joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # 🟢 Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        # Log terminations
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)
