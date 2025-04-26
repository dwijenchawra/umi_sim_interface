import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import scipy.spatial.transform as st

from threading import Lock
from collections import deque
import rclpy

class RosSimInterfaceNode(Node):
    """Node to subscribe to simulated robot and camera data."""
    def __init__(self, num_robots, num_cameras, obs_float32=False, image_transforms=None):
        super().__init__(f'umi_bimanual_sim_subscriber_node_{time.time_ns()}')
        self.cv_bridge = CvBridge()
        self.num_robots = num_robots
        self.num_cameras = num_cameras
        self.obs_float32 = obs_float32
        self.image_transforms = image_transforms or [lambda x: x] * num_cameras

        # Data storage and lock
        self.ros_data_lock = Lock()
        self.camera_buffers = [deque(maxlen=100) for _ in range(self.num_cameras)]
        self.eef_pose_buffers = [deque(maxlen=100) for _ in range(self.num_robots)]
        self.gripper_state_buffers = [deque(maxlen=100) for _ in range(self.num_robots)]
        
        # MAPPING CONVENTION
        # left_arm -> 0
        # right_arm -> 1
        
        # Create subscriptions
        self.camera_subs = []
        for i in range(self.num_cameras):
            topic_name = f'/left_arm/camera' if i == 0 else f'/right_arm/camera'
            self.get_logger().info(f"Subscribing to camera topic: {topic_name}")
            sub = self.create_subscription(
                Image,
                topic_name,
                lambda msg, cam_idx=i: self.camera_callback(msg, cam_idx),
                10)  # QoS profile depth 10
            self.camera_subs.append(sub)

        self.eef_pose_subs = []
        self.gripper_state_subs = []
        
        
        for i in range(self.num_robots):
            arm_prefix = "left_arm" if i == 0 else "right_arm"
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
                Float64,
                grip_topic,
                lambda msg, robot_idx=i: self.gripper_state_callback(msg, robot_idx),
                10)
            self.gripper_state_subs.append(sub_grip)
        
        # create pubs
        # Publishers for sending commands to the simulator
        self.target_pose_pubs = []
        self.target_gripper_pubs = []
        for i in range(self.num_robots):
            arm_prefix = "left_arm" if i == 0 else "right_arm" # Adjust convention as needed
            pub_pose = self.create_publisher(PoseStamped, f'/{arm_prefix}/target_eef_pose', 10)
            self.target_pose_pubs.append(pub_pose)
            pub_grip = self.create_publisher(Float64, f'/{arm_prefix}/target_gripper_width', 10)
            self.target_gripper_pubs.append(pub_grip)


    def _get_ros_time_sec(self, msg) -> float:
        """Extract time in seconds from ROS message header."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def camera_callback(self, msg: Image, camera_idx: int):
        """Handles incoming camera image messages."""
        try:
            recv_time = time.time()
            ros_time_sec = self._get_ros_time_sec(msg)
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_transformed = self.image_transforms[camera_idx](img)
            if self.obs_float32 and img_transformed.dtype == np.uint8:
                img_transformed = img_transformed.astype(np.float32) / 255.0
            img_transformed_chw = np.moveaxis(img_transformed, -1, 0)

            with self.ros_data_lock:
                # print(f"camera_callback: recv data: {recv_time}, ros time: {ros_time_sec}, camera idx: {camera_idx}")
                self.camera_buffers[camera_idx].append(
                    (recv_time, ros_time_sec, img_transformed_chw))
        except Exception as e:
            self.get_logger().error(f"Error processing camera {camera_idx} msg: {e}", throttle_duration_sec=1.0)

    def eef_pose_callback(self, msg: PoseStamped, robot_idx: int):
        """Handles incoming end-effector pose messages."""
        try:
            recv_time = time.time()
            ros_time_sec = self._get_ros_time_sec(msg)
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat_xyzw = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            rotation = st.Rotation.from_quat(quat_xyzw)
            rotvec = rotation.as_rotvec()
            pose_6d = np.concatenate([position, rotvec]).astype(np.float32)

            with self.ros_data_lock:
                # print(f"eef_pose_callback: recv data: {recv_time}, ros time: {ros_time_sec}, robot idx: {robot_idx}")
                self.eef_pose_buffers[robot_idx].append((recv_time, ros_time_sec, pose_6d))
        except Exception as e:
            self.get_logger().error(f"Error processing EEF pose {robot_idx} msg: {e}", throttle_duration_sec=1.0)

    def gripper_state_callback(self, msg: Float64, robot_idx: int):
        """Handles incoming gripper state messages."""
        try:
            recv_time = time.time()
            ros_time_sec = recv_time
            gripper_width = np.array([msg.data], dtype=np.float32)

            with self.ros_data_lock:
                print(f"gripper_state_callback: recv data: {recv_time}, ros time: {ros_time_sec}, robot idx: {robot_idx}, gripper width: {gripper_width}, queue length: {len(self.gripper_state_buffers[robot_idx])}")
                self.gripper_state_buffers[robot_idx].append((recv_time, ros_time_sec, gripper_width))
                print(f"gripper_state_callback: gripper state buffer length: {len(self.gripper_state_buffers[robot_idx])}")
        except Exception as e:
            self.get_logger().error(f"Error processing gripper state {robot_idx} msg: {e}", throttle_duration_sec=1.0)
            
    def get_camera_data(self, camera_topic: str):
        """Get camera data from the buffer."""
        if camera_topic == '/left_arm/camera':
            camera_idx = 0
        elif camera_topic == '/right_arm/camera':
            camera_idx = 1
        
        with self.ros_data_lock:
            print(f"get_camera_data: Camera buffer length: {len(self.camera_buffers[camera_idx])}")
            if len(self.camera_buffers[camera_idx]) > 0:
                recv_time, ros_time_sec, img = self.camera_buffers[camera_idx].popleft()
                return recv_time, ros_time_sec, img
            else:
                return None, None, None
    
    def get_eef_pose(self, robot_idx: int) -> tuple[float, float, np.ndarray]:
        """Get end-effector pose data from the buffer.
        note that pose is in the form of [x, y, z, qx, qy, qz, qw]
        need to convert it into euler angles in here for compatibility with the rest of the codebase
        """
        with self.ros_data_lock:
            # print(f"get_eef_pose: EEF pose buffer length: {len(self.eef_pose_buffers[robot_idx])}")
            if len(self.eef_pose_buffers[robot_idx]) > 0:
                recv_time, ros_time_sec, pose_6d = self.eef_pose_buffers[robot_idx].popleft()
                position = pose_6d[:3]
                rotvec = pose_6d[3:]
                rotation = st.Rotation.from_rotvec(rotvec)
                quat_xyzw = rotation.as_quat()
                pose_6d = np.concatenate([position, quat_xyzw]).astype(np.float32)
                return recv_time, ros_time_sec, pose_6d
            else:
                return None, None, None
            
    def get_gripper_state(self, robot_idx: int):
        """Get gripper state data from the buffer."""
        with self.ros_data_lock:
            print(f"get_gripper_state: Gripper state buffer length: {len(self.gripper_state_buffers[robot_idx])}, robot idx: {robot_idx}")
            if len(self.gripper_state_buffers[robot_idx]) > 0:
                recv_time, ros_time_sec, gripper_width = self.gripper_state_buffers[robot_idx].popleft()
                return recv_time, ros_time_sec, gripper_width
            else:
                return None, None, None
    
    def send_target_pose(self, robot_idx: int, target_pose: np.ndarray):
        pass
    
    def send_target_gripper(self, robot_idx: int, target_width: float):
        msg = Float64()
        msg.data = target_width
        self.target_gripper_pubs[robot_idx].publish(msg)