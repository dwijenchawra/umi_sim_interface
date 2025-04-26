import sys
import os
import pathlib
import threading
import time
import enum
from typing import Optional, List, Dict, Union
import numpy as np
import collections
from multiprocessing.managers import SharedMemoryManager
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.time import Time, Duration
from geometry_msgs.msg import PoseStamped, Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import scipy.interpolate as si
import scipy.spatial.transform as st
from collections import deque

from umi.real_world.multi_camera_visualizer import MultiCameraVisualizer, SimMultiCameraVisualizer
from umi.real_world.video_recorder import VideoRecorder # Keep for potential simulated recording logic
from diffusion_policy.common.timestamp_accumulator import (
    TimestampActionAccumulator,
    ObsAccumulator # Still used for potential data saving logic
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform, optimal_row_cols)

from umi.common.interpolation_util import get_interp1d, PoseInterpolator
from umi.common.pose_util import pose_to_pos_rot, mat_to_pose, pose_to_mat # Keep for internal logic if needed

class Command(enum.Enum):
    """ Commands for the simulation environment control (less relevant now) """
    START_EPISODE = 0
    END_EPISODE = 1
    DROP_EPISODE = 2
    EXEC_ACTIONS = 3 # Command to publish actions

class RosSubscriberNode(Node):
    """Node to subscribe to simulated robot and camera data."""
    def __init__(self, env_instance, num_robots, num_cameras):
        super().__init__(f'umi_bimanual_sim_subscriber_node_{time.time_ns()}')
        self.env_instance: SimulatedBimanualUmiEnv = env_instance
        self.cv_bridge = CvBridge()
        self.num_robots = num_robots
        self.num_cameras = num_cameras

        # Create subscriptions
        self.camera_subs = []
        for i in range(self.num_cameras):
            # Adjust topic names based on your Isaac Sim setup
            topic_name = f'/left_arm/camera' if i == 0 else f'/right_arm/camera'
            self.get_logger().info(f"Subscribing to camera topic: {topic_name}")
            sub = self.create_subscription(
                Image,
                topic_name,
                lambda msg, cam_idx=i: self.camera_callback(msg, cam_idx),
                10) # QoS profile depth 10
            self.camera_subs.append(sub)

        self.eef_pose_subs = []
        self.gripper_state_subs = []
        for i in range(self.num_robots):
            # Adjust topic names based on your Isaac Sim setup
            arm_prefix = "left_arm" if i == 0 else "right_arm" # Example convention
            pose_topic = f'/{arm_prefix}/eef_pose_state'
            grip_topic = f'/{arm_prefix}/gripper_state'
            self.get_logger().info(f"Subscribing to robot {i} pose topic: {pose_topic}")
            self.get_logger().info(f"Subscribing to robot {i} gripper topic: {grip_topic}")

            sub_pose = self.create_subscription(
                PoseStamped,
                pose_topic,
                lambda msg, robot_idx=i: self.eef_pose_callback(msg, robot_idx),
                10)
            self.eef_pose_subs.append(sub_pose)

            sub_grip = self.create_subscription(
                Float64, # Assuming gripper state is published as Float64 (width)
                grip_topic,
                lambda msg, robot_idx=i: self.gripper_state_callback(msg, robot_idx),
                10)
            self.gripper_state_subs.append(sub_grip)

    def _get_ros_time_sec(self, msg) -> float:
        """Extract time in seconds from ROS message header."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def camera_callback(self, msg: Image, camera_idx: int):
        """Handles incoming camera image messages."""
        try:
            recv_time = time.time() # System time when message is received
            ros_time_sec = self._get_ros_time_sec(msg)
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') # Sim might publish BGR
            # img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8') # Or maybe RGB

            # Apply transform immediately if needed before storing
            img_transformed = self.env_instance.image_transforms[camera_idx](img)
            if self.env_instance.obs_float32 and img_transformed.dtype == np.uint8:
                 img_transformed = img_transformed.astype(np.float32) / 255.0
            # Store transformed image (TCHW format often expected by policies)
            # Assuming get_image_transform outputs HWC
            img_transformed_chw = np.moveaxis(img_transformed, -1, 0)

            with self.env_instance.ros_data_lock:
                # Store receive time, ROS time (capture time), and processed image
                self.env_instance.camera_buffers[camera_idx].append(
                    (recv_time, ros_time_sec, img_transformed_chw))
        except Exception as e:
            self.get_logger().error(f"Error processing camera {camera_idx} msg: {e}", throttle_duration_sec=1.0)

    def eef_pose_callback(self, msg: PoseStamped, robot_idx: int):
        """Handles incoming end-effector pose messages."""
        try:
            recv_time = time.time()
            ros_time_sec = self._get_ros_time_sec(msg)
            # Convert PoseStamped to 6D numpy array [pos, rotvec]
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat_xyzw = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            # print(f"Received pose for robot {robot_idx}: {position}, {quat_xyzw}")
            rotation = st.Rotation.from_quat(quat_xyzw)
            rotvec = rotation.as_rotvec()
            pose_6d = np.concatenate([position, rotvec]).astype(np.float32)

            with self.env_instance.ros_data_lock:
                self.env_instance.eef_pose_buffers[robot_idx].append((recv_time, ros_time_sec, pose_6d))
        except Exception as e:
            self.get_logger().error(f"Error processing EEF pose {robot_idx} msg: {e}", throttle_duration_sec=1.0)

    def gripper_state_callback(self, msg: Float64, robot_idx: int):
        """Handles incoming gripper state messages."""
        try:
            recv_time = time.time()
            # Gripper state might not have a timestamp header in simple sims
            ros_time_sec = recv_time # Fallback, ideally use sim time if published
            gripper_width = np.array([msg.data], dtype=np.float32) # Ensure float32 numpy array

            with self.env_instance.ros_data_lock:
                self.env_instance.gripper_state_buffers[robot_idx].append((recv_time, ros_time_sec, gripper_width))
        except Exception as e:
            self.get_logger().error(f"Error processing gripper state {robot_idx} msg: {e}", throttle_duration_sec=1.0)


class SimulatedBimanualUmiEnv:
    """
    ROS2-based simulated environment mimicking the BimanualUmiEnv interface.
    It subscribes to ROS topics for sensor data and publishes commands.
    """
    def __init__(self,
            output_dir,
            # Sim-specific parameters
            num_robots=2,
            num_cameras=2,
            # Env parameters matching the policy's expectations
            frequency=10,
            obs_image_resolution=(224,224), # Target resolution for policy
            max_obs_buffer_size=120, # Increased buffer for sim flexibility
            obs_float32=True, # Policy likely expects float32 images [0,1]
            # Observation horizons
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            # Transforms (keep if policy needs specific input format)
            camera_reorder=None, # Less relevant if topics are named clearly
            no_mirror=False,
            fisheye_converter=None,
            mirror_swap=False,
            # Action execution (less critical for sim, but kept for interface)
            max_pos_speed=2.0, # Sim controller handles limits
            max_rot_speed=6.0, # Sim controller handles limits
            # Recording and visualization (likely handled by simulator)
            enable_multi_cam_vis=True,
            multi_cam_vis_resolution=(1280, 720),
            shm_manager: Optional[SharedMemoryManager]=None, # Needed? Only if using VideoRecorder here.
            # Unused arguments from real env
            **kwargs
            ):

        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        self.video_dir = output_dir.joinpath('videos') # Keep structure for consistency
        self.video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        self.num_robots = num_robots
        self.num_cameras = num_cameras
        self.frequency = frequency
        self.dt = 1.0 / frequency
        self.obs_image_resolution = obs_image_resolution
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_float32 = obs_float32
        self.camera_obs_horizon = camera_obs_horizon
        self.robot_obs_horizon = robot_obs_horizon
        self.gripper_obs_horizon = gripper_obs_horizon
        self.output_dir = output_dir
        self.start_time = None
        self.is_recording = False # Flag to control recording logic

        # ROS2 Setup
        if not rclpy.ok():
            rclpy.init()
            self.node_owner = True # Track if this instance initialized rclpy
        else:
            self.node_owner = False
        # Use a unique node name to avoid conflicts if multiple instances run
        self.node = Node(f'umi_bimanual_sim_env_node_{time.time_ns()}')
        self.ros_data_lock = threading.Lock()

        # Publishers for sending commands to the simulator
        self.target_pose_pubs = []
        self.target_gripper_pubs = []
        for i in range(self.num_robots):
            arm_prefix = "left_arm" if i == 0 else "right_arm" # Adjust convention as needed
            pub_pose = self.node.create_publisher(PoseStamped, f'/{arm_prefix}/target_eef_pose', 10)
            self.target_pose_pubs.append(pub_pose)
            pub_grip = self.node.create_publisher(Float64, f'/{arm_prefix}/target_gripper_width', 10)
            self.target_gripper_pubs.append(pub_grip)

        # Internal buffers for received ROS data (using deques)
        # Store tuples: (receive_time, ros_timestamp_sec, data)
        self.camera_buffers = [collections.deque(maxlen=max_obs_buffer_size) for _ in range(self.num_cameras)]
        self.eef_pose_buffers = [collections.deque(maxlen=max_obs_buffer_size) for _ in range(self.num_robots)]
        self.gripper_state_buffers = [collections.deque(maxlen=max_obs_buffer_size) for _ in range(self.num_robots)]

        # Start the subscriber node in a separate thread
        self.subscriber_node = RosSubscriberNode(self, self.num_robots, self.num_cameras)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.subscriber_node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)

        # Initialize transforms (assuming sim output is BGR HWC uint8)
        self.image_transforms = [
            get_image_transform(
                input_res=(1280, 720), # Adjust based on actual sim camera resolution
                output_res=self.obs_image_resolution,
                bgr_to_rgb=True) # Policy expects RGB
            for _ in range(self.num_cameras)
        ]

        # Initialize action/obs accumulators if saving sim data to replay buffer
        self.obs_accumulator = None
        self.action_accumulator = None

        # Align camera index (can be fixed for sim)
        self.align_camera_idx = 0 # Typically use camera 0 as reference
        
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(self.camera_buffers),
            in_wh_ratio=4/3,
            max_resolution=multi_cam_vis_resolution
        )

        multi_cam_vis = None
        if enable_multi_cam_vis:
            multi_cam_vis = SimMultiCameraVisualizer(
                env_instance=self,
                row=row,
                col=col,
                rgb_to_bgr=False
            )
        self.multi_cam_vis = multi_cam_vis


        print("[SimulatedBimanualUmiEnv] Initialized.")

    # ======== start-stop API =============
    def start(self, wait=True):
        """Starts the ROS2 node executor thread."""
        if not self.spin_thread.is_alive():
            self.spin_thread.start()
            # Add a small delay to allow subscriptions to potentially establish
            time.sleep(1.0)
            print("[SimulatedBimanualUmiEnv] ROS2 spinning started.")
            
            # start multi camera visualizer if enabled
            if self.multi_cam_vis is not None:
                self.multi_cam_vis.start(wait=wait)
                print("[SimulatedBimanualUmiEnv] Multi camera visualizer started.")
        else:
             print("[SimulatedBimanualUmiEnv] ROS2 spinning already started.")


    def stop(self, wait=True):
        """Stops the ROS2 node executor and potentially shuts down rclpy."""
        self.end_episode() # Ensure data is saved if recording
        print("[SimulatedBimanualUmiEnv] Stopping ROS2 executor...")
        # Shutdown executor first to stop callbacks
        if self.executor is not None:
            self.executor.shutdown()
            self.executor = None # Prevent double shutdown
        if self.spin_thread.is_alive():
             self.spin_thread.join(timeout=1.0) # Wait shortly for thread to exit

        print("[SimulatedBimanualUmiEnv] Stopping ROS2 node...")
        if self.node is not None:
             self.node.destroy_node()
             self.node = None # Prevent double destruction
        if self.subscriber_node is not None:
             self.subscriber_node.destroy_node()
             self.subscriber_node = None
        if self.node_owner and rclpy.ok():
             rclpy.try_shutdown() # Use try_shutdown if rclpy might be shared
        print("[SimulatedBimanualUmiEnv] Stopped.")


    @property
    def is_ready(self):
        """Checks if essential data has been received."""
        with self.ros_data_lock:
            # Check for *recent* data in buffers
            min_data_points = 1 # Need at least one data point
            now = time.time()
            max_age = 5.0 # seconds - adjust as needed

            cameras_ready = all(
                len(buf) >= min_data_points and (now - buf[-1][0]) < max_age
                for buf in self.camera_buffers if len(buf) > 0 # Check only non-empty first
            ) and all(len(buf)>0 for buf in self.camera_buffers) # Ensure all have *some* data

            poses_ready = all(
                len(buf) >= min_data_points and (now - buf[-1][0]) < max_age
                for buf in self.eef_pose_buffers if len(buf) > 0
            ) and all(len(buf)>0 for buf in self.eef_pose_buffers)

            grippers_ready = all(
                len(buf) >= min_data_points and (now - buf[-1][0]) < max_age
                for buf in self.gripper_state_buffers if len(buf) > 0
            ) and all(len(buf)>0 for buf in self.gripper_state_buffers)

        # Print status for debugging
        if not (cameras_ready and poses_ready and grippers_ready):
            print(f"Readiness check: Cam={cameras_ready}, Pose={poses_ready}, Grip={grippers_ready}")

        return cameras_ready and poses_ready and grippers_ready

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        """
        Fetches data from internal buffers and performs timestamp alignment using ROS timestamps.
        """
        if not self.is_ready:
            # print("[SimulatedBimanualUmiEnv] Warning: Env not ready, returning empty obs.")
            return {} # Or raise an error, or return last known obs

        obs_data = {'timestamp': None} # Initialize with timestamp key
        last_timestamp = -1.0

        with self.ros_data_lock:
            # --- Find the latest valid reference timestamp ---
            # Use the chosen alignment camera's latest ROS timestamp
            ref_cam_buffer = list(self.camera_buffers[self.align_camera_idx])
            if not ref_cam_buffer:
                print(f"[SimulatedBimanualUmiEnv] Warning: Reference camera {self.align_camera_idx} buffer empty.")
                return {} # Cannot align without reference
            last_timestamp = ref_cam_buffer[-1][1] # ROS time is index 1

            # --- Calculate target timestamps based on frequency and horizons ---
            target_camera_timestamps = last_timestamp - (
                np.arange(self.camera_obs_horizon)[::-1] * self.dt)
            target_robot_timestamps = last_timestamp - (
                np.arange(self.robot_obs_horizon)[::-1] * self.dt)
            target_gripper_timestamps = last_timestamp - (
                np.arange(self.gripper_obs_horizon)[::-1] * self.dt)

            # --- Align Camera Data ---
            all_cam_aligned = True
            for i in range(self.num_cameras):
                cam_buffer = list(self.camera_buffers[i]) # Copy for safe iteration
                if not cam_buffer:
                    all_cam_aligned = False
                    print(f"[SimulatedBimanualUmiEnv] Warning: Camera {i} buffer empty.")
                    break

                cam_times_ros = np.array([t[1] for t in cam_buffer]) # ROS time
                cam_imgs_processed = [t[2] for t in cam_buffer] # Already processed CHW images

                aligned_imgs = []
                for t_target in target_camera_timestamps:
                    # Find nearest image based on ROS time
                    idx = np.argmin(np.abs(cam_times_ros - t_target))
                    aligned_imgs.append(cam_imgs_processed[idx])
                    
                ##### IMAGES ARE IN WHC FORMAT : CONVERT TO CHW NOW
                aligned_imgs = [np.transpose(img, (1, 2, 0)) for img in aligned_imgs]
                
                if aligned_imgs:
                     # Stack to TCHW
                    obs_data[f'camera{i}_rgb'] = np.stack(aligned_imgs)
                else:
                     all_cam_aligned = False
                     print(f"[SimulatedBimanualUmiEnv] Warning: Could not align images for camera {i}.")
                     break # Stop alignment if one camera fails

            if not all_cam_aligned:
                return {} # Return empty if camera alignment failed

            # Update the main timestamp array for the obs dict
            obs_data['timestamp'] = target_camera_timestamps

            # --- Align Robot Pose Data ---
            all_robots_aligned = True
            for i in range(self.num_robots):
                robot_buffer = list(self.eef_pose_buffers[i])
                if not robot_buffer:
                    all_robots_aligned = False
                    print(f"[SimulatedBimanualUmiEnv] Warning: Robot {i} pose buffer empty.")
                    break

                robot_times_ros = np.array([t[1] for t in robot_buffer])
                robot_poses = np.stack([t[2] for t in robot_buffer])

                try:
                    # Interpolate using ROS timestamps
                    pose_interp = PoseInterpolator(t=robot_times_ros, x=robot_poses)
                    aligned_poses = pose_interp(target_robot_timestamps)
                    obs_data[f'robot{i}_eef_pos'] = aligned_poses[:, :3].astype(np.float32)
                    obs_data[f'robot{i}_eef_rot_axis_angle'] = aligned_poses[:, 3:].astype(np.float32)
                except ValueError as e:
                    # Handle cases where interpolation fails (e.g., duplicate timestamps)
                    print(f"[SimulatedBimanualUmiEnv] Warning: Robot {i} pose interpolation failed: {e}")
                    # Fallback: use nearest neighbor for all steps
                    aligned_poses_nn = []
                    for t_target in target_robot_timestamps:
                        idx = np.argmin(np.abs(robot_times_ros - t_target))
                        aligned_poses_nn.append(robot_poses[idx])
                    aligned_poses = np.stack(aligned_poses_nn)
                    obs_data[f'robot{i}_eef_pos'] = aligned_poses[:, :3].astype(np.float32)
                    obs_data[f'robot{i}_eef_rot_axis_angle'] = aligned_poses[:, 3:].astype(np.float32)


            if not all_robots_aligned:
                return {} # Return empty if robot alignment failed


            # --- Align Gripper Data ---
            all_grippers_aligned = True
            for i in range(self.num_robots):
                gripper_buffer = list(self.gripper_state_buffers[i])
                if not gripper_buffer:
                    all_grippers_aligned = False
                    print(f"[SimulatedBimanualUmiEnv] Warning: Gripper {i} buffer empty.")
                    break

                gripper_times_ros = np.array([t[1] for t in gripper_buffer])
                gripper_widths = np.stack([t[2] for t in gripper_buffer])

                try:
                    # Interpolate using ROS timestamps
                    gripper_interp = get_interp1d(t=gripper_times_ros, x=gripper_widths)
                    aligned_widths = gripper_interp(target_gripper_timestamps)
                    obs_data[f'robot{i}_gripper_width'] = aligned_widths.astype(np.float32)
                except ValueError as e:
                     # Handle cases where interpolation fails (e.g., duplicate timestamps)
                    print(f"[SimulatedBimanualUmiEnv] Warning: Gripper {i} interpolation failed: {e}")
                    # Fallback: use nearest neighbor
                    aligned_widths_nn = []
                    for t_target in target_gripper_timestamps:
                        idx = np.argmin(np.abs(gripper_times_ros - t_target))
                        aligned_widths_nn.append(gripper_widths[idx])
                    aligned_widths = np.stack(aligned_widths_nn)
                    obs_data[f'robot{i}_gripper_width'] = aligned_widths.astype(np.float32)

            if not all_grippers_aligned:
                return {} # Return empty if gripper alignment failed


        # Accumulate raw data for replay buffer saving (if enabled)
        # This part needs careful thought: what raw data to save?
        # Maybe save the *buffered* data before alignment?
        if self.obs_accumulator is not None:
             # This part is complex and depends on what state representation
             # is desired in the replay buffer vs. what the policy needs.
             # For now, we'll skip detailed implementation here.
             pass

        return obs_data

    def exec_actions(self,
            actions: np.ndarray,
            timestamps: np.ndarray, # Expected to be target execution times in time.time() format
            compensate_latency=False): # Ignored in sim

        """Publishes actions to the appropriate ROS topics."""
        if not self.is_ready:
            print("[SimulatedBimanualUmiEnv] Warning: Env not ready, skipping action execution.")
            return

        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)

        assert actions.shape[1] == 7 * self.num_robots

        now_ros_time = self.node.get_clock().now()
        now_time_sec = time.time()

        for i in range(len(actions)):
            # print(f"Executing action {i:02d} at timestamp {timestamps[i]:.3f}: {[f'{x:.2f}' for x in actions[i]]}")
            action_target_time_sec = timestamps[i]
            # Calculate the ROS Time corresponding to the target execution time
            # This assumes a relatively stable offset between time.time() and ROS time
            # A more robust solution might involve explicitly syncing clocks or using ROS durations
            time_diff_sec = action_target_time_sec - now_time_sec
            target_ros_time = now_ros_time + Duration(seconds=time_diff_sec)

            # Iterate through robots
            for robot_idx in range(self.num_robots):
                # Extract pose and gripper action for this robot
                pose_6d = actions[i, robot_idx*7 : robot_idx*7 + 6]
                gripper_width = actions[i, robot_idx*7 + 6]

                # Create PoseStamped message
                pose_msg = PoseStamped()
                pose_msg.header.stamp = target_ros_time.to_msg()
                # Frame ID should match what the simulator controller expects (e.g., 'world' or 'base_link')
                pose_msg.header.frame_id = "world"
                pose_msg.pose.position.x = float(pose_6d[0])
                pose_msg.pose.position.y = float(pose_6d[1])
                pose_msg.pose.position.z = float(pose_6d[2])
                # Convert rotvec to quaternion (assuming scipy format)
                quat_xyzw = st.Rotation.from_rotvec(pose_6d[3:]).as_quat()
                pose_msg.pose.orientation.x = float(quat_xyzw[0])
                pose_msg.pose.orientation.y = float(quat_xyzw[1])
                pose_msg.pose.orientation.z = float(quat_xyzw[2])
                pose_msg.pose.orientation.w = float(quat_xyzw[3])
                

                # Create Float64 message for gripper
                gripper_msg = Float64()
                gripper_msg.data = float(gripper_width)

                # Publish
                if robot_idx < len(self.target_pose_pubs) and robot_idx < len(self.target_gripper_pubs):
                    self.target_pose_pubs[robot_idx].publish(pose_msg)
                    self.target_gripper_pubs[robot_idx].publish(gripper_msg)
                else:
                    self.node.get_logger().warn(f"Publisher index {robot_idx} out of bounds.")

        # Record actions if needed (optional for sim)
        if self.action_accumulator is not None:
             # Use target execution timestamps (from input `timestamps`)
             self.action_accumulator.put(actions, timestamps)

    # recording API (Simplified for simulation context)
    def start_episode(self, start_time=None):
        "Marks the start of an episode for potential data logging."
        if start_time is None:
            start_time = time.time() # Use system time as reference start
        self.start_time = start_time
        self.is_recording = True # Set recording flag

        # Reset accumulators if used for saving data
        if self.replay_buffer is not None: # Check if replay buffer exists
            self.obs_accumulator = ObsAccumulator() # Recreate if needed
            self.action_accumulator = TimestampActionAccumulator(
                start_time=self.start_time,
                dt=1/self.frequency)

        episode_id = self.replay_buffer.n_episodes if self.replay_buffer else -1
        print(f'Simulated Episode {episode_id} started at {start_time}!')

    def end_episode(self):
        "Finalizes episode data saving if recording was active."
        if not self.is_recording:
            return

        print(f'Simulated Episode {self.replay_buffer.n_episodes} stopped.')
        self.is_recording = False

        if self.obs_accumulator is not None and self.action_accumulator is not None:
            # Logic to save accumulated data to replay buffer
            # This needs careful implementation based on what exactly needs to be saved
            # and how to align the ObsAccumulator data (which might store raw, high-freq
            # poses) with the action accumulator timestamps.
            # Example sketch (needs refinement):
            action_timestamps = self.action_accumulator.timestamps
            actions = self.action_accumulator.actions
            if len(action_timestamps) > 0:
                episode_data = {'timestamp': action_timestamps, 'action': actions}
                # Interpolate obs data from ObsAccumulator to match action_timestamps
                # ... (complex interpolation logic similar to original end_episode) ...
                # self.replay_buffer.add_episode(episode_data, compressors='disk')
                # print(f"Simulated Episode {self.replay_buffer.n_episodes - 1} data processed for saving.")
                pass # Placeholder for data saving logic

            self.obs_accumulator = None
            self.action_accumulator = None


    def drop_episode(self):
        """Discards the current episode data."""
        self.is_recording = False # Stop recording flag
        # If an episode was actually added to the buffer, remove it
        if self.replay_buffer is not None and self.replay_buffer.is_episode_open():
             self.replay_buffer.drop_episode()
             print(f'Simulated Episode {self.replay_buffer.n_episodes} dropped!')
        else:
             print("No active episode in replay buffer to drop.")

        # Clear accumulators
        self.obs_accumulator = None
        self.action_accumulator = None

    def get_robot_state(self):
        """Returns the latest known robot state from buffers."""
        # Returns a list of dicts, one per robot
        states = list()
        with self.ros_data_lock:
            for i in range(self.num_robots):
                 if len(self.eef_pose_buffers[i]) > 0:
                     latest_pose = self.eef_pose_buffers[i][-1][2] # index 2 is pose data
                     # Mimic the structure of the original state dict if needed
                     states.append({'TargetTCPPose': latest_pose, 'ActualTCPPose': latest_pose})
                 else:
                     # Return default/dummy state if no data received yet
                     states.append({'TargetTCPPose': np.zeros(6), 'ActualTCPPose': np.zeros(6)})
        return states

    def get_gripper_state(self):
        """Returns the latest known gripper state from buffers."""
        # Returns a list of dicts, one per gripper
        states = list()
        with self.ros_data_lock:
             for i in range(self.num_robots):
                  if len(self.gripper_state_buffers[i]) > 0:
                       latest_width = self.gripper_state_buffers[i][-1][2] # index 2 is width data
                       # Mimic the structure if needed
                       states.append({'gripper_position': latest_width[0]}) # Assuming width is first element
                  else:
                       states.append({'gripper_position': 0.0}) # Default/dummy state
        return states