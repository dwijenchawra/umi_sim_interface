"""
Usage:
(umi): python eval_real.py -i <ckpt_path> -o <save_dir> --robot_config <config.yaml> [--sim]

# To run with simulator:
(umi): python eval_real.py -i <ckpt_path> -o <save_dir> --robot_config example/eval_robots_config.yaml --sim

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.
Press "M" to move to the start pose of the matched episode.
Press "E/W" to cycle through matched episodes.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly!

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import inspect
import sys
import os
import traceback

import matplotlib.image


# from umi.real_world.bimanual_umi_sim_env import SimulatedBimanualUmiEnv
from umi.sim.bimanual_umi_sim_env_new import SimulatedBimanualUmiEnv

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
# Conditional import based on --sim flag
# from umi.real_world.bimanual_umi_env import BimanualUmiEnv # Original
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
# from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose
import rclpy # Import rclpy
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver("eval", eval, replace=True)

# ... (solve_table_collision and solve_sphere_collision remain the same) ...
def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
@click.option('--sim', is_flag=True, default=False, help="Run in simulation mode using ROS2.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False) # Kept for potential sim use
@click.option('-sf', '--sim_fov', type=float, default=None) # Kept for potential sim use
@click.option('-ci', '--camera_intrinsics', type=str, default=None) # Kept for potential sim use
@click.option('--mirror_swap', is_flag=True, default=False) # Kept for potential sim use
def main(input, output, robot_config,
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints,
    steps_per_inference, max_duration,
    frequency, command_latency,
    sim, no_mirror, sim_fov, camera_intrinsics, mirror_swap):


    # rclpy.init() # Initialize ROS2 Python client library
    # this happens in the env

    max_gripper_width = 0.09
    gripper_speed = 0.2

    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))

    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    num_robots = len(robots_config)
    num_cameras = len(camera_reorder) # Assume one camera per robot + potentially others

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)

    try:
        with SharedMemoryManager() as shm_manager:
            with KeystrokeCounter() as key_counter, \
                SimulatedBimanualUmiEnv(
                    output_dir=output,
                    # ros_node=None,  # Add appropriate RosDataNode instance here
                    frequency=frequency,
                    camera_obs_latency=cfg.task.camera_obs_latency,
                    robot_obs_latency=0,  # Default value
                    gripper_obs_latency=0,  # Default value
                    # camera_down_sample_steps=1,  # Uncomment if needed
                    # robot_down_sample_steps=1,  # Uncomment if needed
                    # gripper_down_sample_steps=1,  # Uncomment if needed
                    camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                    robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                    gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                    obs_image_resolution=obs_res,
                    max_pos_speed=2.0,
                    max_rot_speed=6.0,
                    # robot_action_latency=0,  # Uncomment if needed
                    # gripper_action_latency=0,  # Uncomment if needed
                    enable_multi_cam_vis=True,
                    # multi_cam_vis_resolution=(960, 960),  # Uncomment if needed
                    shm_manager=shm_manager
                ) as env:
                    
                # SimulatedBimanualUmiEnv(
                #     output_dir=output,
                #     frequency=frequency,
                #     obs_image_resolution=obs_res,
                #     # obs_floa
                #     # t32=True,
                #     # camera_reorder=[int(x) for x in camera_reorder],
                #     # init_joints=init_joints,
                #     enable_multi_cam_vis=True,
                #     camera_obs_latency=cfg.task.camera_obs_latency,
                #     camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                #     robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                #     gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                #     no_mirror=no_mirror,
                #     fisheye_converter=fisheye_converter,
                #     mirror_swap=mirror_swap,
                #     max_pos_speed=2.0,
                #     max_rot_speed=6.0,
                #     num_robots=num_robots,
                #     num_cameras=num_cameras,
                #     shm_manager=shm_manager
                # ) as env:



                # cv2.setNumThreads(2) # Limit threads for OpenCV as well
                print("Waiting for env readiness...")
                while not env.is_ready: # Wait for env (and ROS connections if sim)
                    time.sleep(0.1)
                print("Env is ready.")

                # load match_dataset
                episode_first_frame_map = dict()
                match_replay_buffer = None
                if match_dataset is not None:
                    match_dir = pathlib.Path(match_dataset)
                    match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                    match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                    match_video_dir = match_dir.joinpath('videos')
                    for vid_dir in match_video_dir.glob("*/"):
                        episode_idx = int(vid_dir.stem)
                        match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                        if match_video_path.exists():
                            img = None
                            try:
                                with av.open(str(match_video_path)) as container:
                                    stream = container.streams.video[0]
                                    for frame in container.decode(stream):
                                        img = frame.to_ndarray(format='rgb24')
                                        break
                            except Exception as e:
                                print(f"Error loading video {match_video_path}: {e}")
                            if img is not None:
                                episode_first_frame_map[episode_idx] = img
                print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

                # creating model
                policy_cls = hydra.utils.get_class(cfg._target_)
                workspace = policy_cls(cfg)
                workspace: BaseWorkspace
                workspace.load_payload(payload, exclude_keys=None, include_keys=None)

                policy = workspace.model
                if cfg.training.use_ema:
                    policy = workspace.ema_model
                policy.num_inference_steps = 16 # DDIM inference iterations
                obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
                action_pose_repr = cfg.task.pose_repr.action_pose_repr
                print('obs_pose_rep', obs_pose_rep)
                print('action_pose_repr', action_pose_repr)

                device = torch.device('cuda')
                policy.eval().to(device)

                print("Warming up policy inference")
                # Need to ensure env is ready and provides an initial obs
                # for _ in range(10): # Retry loop
                #      if env.is_ready:
                obs = env.get_obs()
                #           if obs and 'timestamp' in obs: # Check if obs is valid
                #                break
                #      print("Env not ready or initial obs not available, waiting...")
                #      time.sleep(0.5)
                # else:
                #      print("Error: Env failed to become ready or provide initial observation.")
                #      exit(1)

                episode_start_pose = list()
                for robot_id in range(num_robots):
                    pose = np.concatenate([
                        obs[f'robot{robot_id}_eef_pos'],
                        obs[f'robot{robot_id}_eef_rot_axis_angle']
                    ], axis=-1)[-1]
                    episode_start_pose.append(pose)
                with torch.no_grad():
                    policy.reset()
                    obs_dict_np = get_real_umi_obs_dict(
                        env_obs=obs, shape_meta=cfg.task.shape_meta,
                        obs_pose_repr=obs_pose_rep,
                        tx_robot1_robot0=tx_robot1_robot0,
                        episode_start_pose=episode_start_pose)
                    obs_dict = dict_apply(obs_dict_np,
                        lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                    result = policy.predict_action(obs_dict)
                    action = result['action_pred'][0].detach().to('cpu').numpy()
                    
                    
                    print(f"Action shape: {action.shape}, Expected shape: {10 * num_robots}")
                    assert action.shape[-1] == 10 * num_robots, f"Action shape mismatch: {action.shape[-1]} != {10 * num_robots}"
                    action = get_real_umi_action(action, obs, action_pose_repr)
                    assert action.shape[-1] == 7 * num_robots
                    del result

                print('Ready!')
                # while True:
                #     # ========= human control loop ==========
                #     print("Human in control!")
                #     robot_states = env.get_robot_state()
                #     # Handle potential None during init for sim env
                #     target_pose = np.stack([rs.get('TargetTCPPose', np.zeros(6)) for rs in robot_states])

                #     gripper_states = env.get_gripper_state()
                #     gripper_target_pos = np.asarray([gs.get('gripper_position', max_gripper_width) for gs in gripper_states])

                #     control_robot_idx_list = [0]

                #     t_start = time.monotonic()
                #     iter_idx = 0
                #     while True:
                #         # calculate timing
                #         t_cycle_end = t_start + (iter_idx + 1) * dt
                #         t_sample = t_cycle_end - command_latency
                #         t_command_target = t_cycle_end + dt

                #         # pump obs
                #         obs = env.get_obs()
                #         if not obs: # Skip if obs is not ready yet
                #             precise_wait(t_cycle_end)
                #             iter_idx += 1
                #             continue

                #         # visualize
                #         episode_id = env.replay_buffer.n_episodes
                #         # Use a camera guaranteed to exist
                #         vis_img_key = f'camera{vis_camera_idx}_rgb'
                #         if vis_img_key not in obs:
                #              vis_img_key = next(iter(k for k in obs.keys() if k.endswith('_rgb')))

                #         vis_img = obs[vis_img_key][-1].copy() # Use copy

                #         match_episode_id = episode_id
                #         if match_episode is not None:
                #             match_episode_id = match_episode
                #         if match_episode_id in episode_first_frame_map:
                #             match_img = episode_first_frame_map[match_episode_id]
                #             ih, iw, _ = match_img.shape
                #             oh, ow, _ = vis_img.shape
                #             tf = get_image_transform(
                #                 input_res=(iw, ih),
                #                 output_res=(ow, oh),
                #                 bgr_to_rgb=False) # Assume RGB from sim/obs
                #             match_img = tf(match_img).astype(np.float32) / 255.0
                #             vis_img = np.clip((vis_img + match_img) / 2.0, 0, 1) # Clip for safety

                #         # Prepare other camera views for concatenation
                #         cam_views = []
                #         for i in range(env.num_cameras): # Use num_cameras from env
                #             cam_key = f'camera{i}_rgb'
                #             if cam_key in obs:
                #                 cam_views.append(obs[cam_key][-1])
                #             else:
                #                 # Add placeholder if a camera view is missing
                #                 cam_views.append(np.zeros_like(vis_img)) 
                                
                #         # Ensure all images have the same height before concatenating
                #         target_h = vis_img.shape[0]
                #         processed_views = []
                #         for img in cam_views:
                #              if img.shape[0] != target_h:
                #                   scale = target_h / img.shape[0]
                #                   new_w = int(img.shape[1] * scale)
                #                   img = cv2.resize(img, (new_w, target_h))
                #              processed_views.append(img)

                #         # Concatenate available views
                #         if processed_views:
                #              final_vis_img = np.concatenate(processed_views, axis=1)
                #         else:
                #              final_vis_img = vis_img # Fallback if no views were processed


                #         text = f'Episode: {episode_id}'
                #         cv2.putText(
                #             final_vis_img, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5, lineType=cv2.LINE_AA, thickness=3, color=(0,0,0))
                #         cv2.putText(
                #             final_vis_img, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #             fontScale=0.5, thickness=1, color=(255,255,255))

                #         cv2.imshow('default', (final_vis_img[...,::-1]*255).astype(np.uint8)) # Convert to BGR uint8
                #         _ = cv2.pollKey()
                #         press_events = key_counter.get_press_events()
                #         start_policy = False
                #         for key_stroke in press_events:
                #             if key_stroke == KeyCode(char='q'):
                #                 env.end_episode()
                #                 exit(0)
                #             elif key_stroke == KeyCode(char='c'):
                #                 start_policy = True
                #             elif key_stroke == KeyCode(char='e'):
                #                 if match_episode is not None:
                #                     match_episode = min(match_episode + 1, env.replay_buffer.n_episodes - 1)
                #             elif key_stroke == KeyCode(char='w'):
                #                 if match_episode is not None:
                #                     match_episode = max(match_episode - 1, 0)
                #             elif key_stroke == KeyCode(char='m'):
                #                 if match_replay_buffer is not None:
                #                     try:
                #                         duration = 3.0
                #                         ep = match_replay_buffer.get_episode(match_episode_id)
                #                         target_pose_list = []
                #                         gripper_target_pos_list = []
                #                         for robot_idx in range(num_robots):
                #                             pos = ep[f'robot{robot_idx}_eef_pos'][0]
                #                             rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                #                             grip = ep[f'robot{robot_idx}_gripper_width'][0]
                #                             pose = np.concatenate([pos, rot])
                #                             target_pose_list.append(pose)
                #                             gripper_target_pos_list.append(grip)

                #                         # Send commands via exec_actions (publishes to ROS)
                #                         action_teleop = np.zeros((7 * num_robots,))
                #                         for robot_idx in range(num_robots):
                #                              action_teleop[robot_idx*7 : robot_idx*7 + 6] = target_pose_list[robot_idx]
                #                              action_teleop[robot_idx*7 + 6] = gripper_target_pos_list[robot_idx]

                #                         # Publish target pose/gripper for duration
                #                         t_move_end = time.time() + duration
                #                         while time.time() < t_move_end:
                #                              env.exec_actions(
                #                                 actions=[action_teleop], # Send as a single step action list
                #                                 timestamps=[time.time() + dt], # Target next step
                #                                 compensate_latency=False)
                #                              time.sleep(dt / 2) # Sleep briefly

                #                         # Update local state after move
                #                         target_pose = np.stack(target_pose_list)
                #                         gripper_target_pos = np.array(gripper_target_pos_list)

                #                     except Exception as e:
                #                         print(f"Error moving to matched pose: {e}")
                #             elif key_stroke == Key.backspace:
                #                 if click.confirm('Are you sure to drop an episode?'):
                #                     env.drop_episode()
                #                     key_counter.clear()
                #             elif key_stroke == KeyCode(char='a'):
                #                 control_robot_idx_list = list(range(num_robots))
                #             elif key_stroke == KeyCode(char='1'):
                #                 control_robot_idx_list = [0]
                #             elif key_stroke == KeyCode(char='2'):
                #                 if num_robots > 1:
                #                     control_robot_idx_list = [1]

                #         if start_policy:
                #             break

                #         precise_wait(t_sample)
                #         # get teleop command


                #         dpos_grip = 0
                #         for robot_idx in control_robot_idx_list:
                #             gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos_grip, 0, max_gripper_width)

                #         # Apply collision avoidance (can be kept if needed, adjust based on sim setup)
                #         if not sim: # Only apply for real world
                #             for robot_idx in control_robot_idx_list:
                #                 solve_table_collision(
                #                     ee_pose=target_pose[robot_idx],
                #                     gripper_width=gripper_target_pos[robot_idx],
                #                     height_threshold=robots_config[robot_idx]['height_threshold'])
                #             if num_robots > 1:
                #                 solve_sphere_collision(
                #                     ee_poses=target_pose,
                #                     robots_config=robots_config)

                #         action_teleop = np.zeros((7 * num_robots,))
                #         for robot_idx in range(num_robots):
                #              action_teleop[robot_idx*7 : robot_idx*7 + 6] = target_pose[robot_idx]
                #              action_teleop[robot_idx*7 + 6] = gripper_target_pos[robot_idx]

                #         # execute teleop command via ROS
                #         env.exec_actions(
                #             actions=[action_teleop], # Send as a single step action list
                #             timestamps=[time.time() + dt] # Target next step's time
                #         )
                #         precise_wait(t_cycle_end)
                #         iter_idx += 1

                # ========== policy control loop ==============
                                
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    # env.start_episode(eval_t_start) # Env handles recording start internally if needed

                    # Get initial pose for relative calculations if needed by policy
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(num_robots):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)


                    # wait for observation pipeline to be ready
                    frame_latency = 1/60 # Small wait to ensure latest obs is used
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Policy Started!")
                    iter_idx = 0
                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                        # get obs
                        obs = env.get_obs()
                        if not obs or 'timestamp' not in obs:
                            print("Waiting for observation...")
                            precise_wait(t_cycle_end - frame_latency) # Wait and retry
                            iter_idx += steps_per_inference
                            continue

                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            # Prepare obs for policy (using the same function)
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta,
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0, # Pass relative tf if needed
                                episode_start_pose=episode_start_pose) # Pass start pose if needed
                            obs_dict = dict_apply(obs_dict_np,
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            # Convert policy action to environment action (e.g., absolute poses)
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)
                            print('Inference latency:', time.time() - s)

                        # Apply collision avoidance to policy actions (optional for sim, but good practice)
                        this_target_poses = action.reshape(-1, num_robots, 7) # T, NRobots, 7
                        for t_idx in range(this_target_poses.shape[0]):
                             target_pose_step = this_target_poses[t_idx] # NR, 7
                             if not sim:
                                 for robot_idx in range(num_robots):
                                      solve_table_collision(
                                          ee_pose=target_pose_step[robot_idx, :6],
                                          gripper_width=target_pose_step[robot_idx, 6],
                                          height_threshold=robots_config[robot_idx]['height_threshold']
                                      )
                                 if num_robots > 1:
                                      solve_sphere_collision(
                                          ee_poses=target_pose_step, # Pass NR, 7 array
                                          robots_config=robots_config
                                      )
                        action = this_target_poses.reshape(action.shape[0], -1) # Reshape back T, (NR*7)


                        # --- Timing and Action Execution ---
                        action_timestamps = (np.arange(len(action), dtype=np.float64)
                            ) * dt + obs_timestamps[-1] # Target times relative to last obs

                        # Execute actions via ROS
                        env.exec_actions(
                            actions=action,
                            timestamps=action_timestamps,
                            compensate_latency=False # Let sim handle timing via ROS messages
                        )
                        print(f"Submitted {len(action)} steps of actions.")
                        # --- End Timing and Action Execution ---


                        # # visualize
                        # vis_img_key = f'camera{vis_camera_idx}_rgb'
                        # if vis_img_key not in obs:
                        #      vis_img_key = next(iter(k for k in obs.keys() if k.endswith('_rgb')))
                        # vis_img = obs[vis_img_key][-1]
                        # episode_id = env.replay_buffer.n_episodes
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img, text, (10,20), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5, thickness=1, color=(255,255,255))
                        # cv2.imshow('default', (vis_img[...,::-1]*255).astype(np.uint8)) # Convert to BGR uint8

                        # key_stroke = cv2.pollKey()
                        # stop_episode = False
                        # if key_stroke == ord('s'):
                        #     print('Stopped.')
                        #     stop_episode = True
                        # press_events = key_counter.get_press_events()
                        # for key_stroke_event in press_events:
                        #     if key_stroke_event == KeyCode(char='s'):
                        #         print('Stopped.')
                        #         stop_episode = True

                        # t_since_start = time.time() - eval_t_start
                        # if t_since_start > max_duration:
                        #     print("Max Duration reached.")
                        #     stop_episode = True
                        # if stop_episode:
                        #     env.end_episode()
                        #     break

                        # wait for execution cycle
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += steps_per_inference

                except KeyboardInterrupt:
                    print("Interrupted!")
                    env.end_episode()
                except Exception as e:
                    print(f"Error during policy execution: {e}")
                    traceback.print_exc()

                    env.end_episode()

                print("Policy control stopped.")

    except Exception as e:
         print(f"An error occurred in the main loop: {e}")
         traceback.print_exc()
    except KeyboardInterrupt:
        print("Keyboard interrupt detected.")
        sys.exit(0)
    finally:
         # Ensure ROS is shutdown properly
         if rclpy.ok():
              rclpy.shutdown()
         print("ROS shutdown.")

# %%
if __name__ == '__main__':
    main()