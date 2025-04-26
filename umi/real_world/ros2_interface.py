# --- START OF umi/real_world/ros2_interface.py ---
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Float64
import numpy as np
import threading
from typing import Dict, List, Optional, Tuple

class ROS2SimInterface:
    def __init__(self, node_name="umi_ros2_sim_interface"):
        if not rclpy.ok():
            rclpy.init()
        self.node = Node(node_name)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)

        # Store latest messages
        self._latest_joint_state_left: Optional[JointState] = None
        self._latest_joint_state_right: Optional[JointState] = None
        self._latest_eef_pose_left: Optional[PoseStamped] = None
        self._latest_eef_pose_right: Optional[PoseStamped] = None
        self._latest_image_left: Optional[Image] = None
        self._latest_image_right: Optional[Image] = None
        self._lock = threading.Lock()

        # QoS Profiles
        qos_profile_reliable_volatile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )
        qos_profile_best_effort_volatile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.VOLATILE
        )

        # Publishers
        self.eef_command_publisher_left = self.node.create_publisher(
            PoseStamped, "/left_arm/eef_command", qos_profile_reliable_volatile)
        self.eef_command_publisher_right = self.node.create_publisher(
            PoseStamped, "/right_arm/eef_command", qos_profile_reliable_volatile)
        self.gripper_command_publisher_left = self.node.create_publisher(
            Float64, "/left_arm/gripper_command", qos_profile_reliable_volatile)
        self.gripper_command_publisher_right = self.node.create_publisher(
            Float64, "/right_arm/gripper_command", qos_profile_reliable_volatile)

        # Subscribers
        self.joint_state_subscriber_left = self.node.create_subscription(
            JointState, "/left_arm/joint_states", self._joint_state_callback_left, 10) # Default QoS is usually fine
        self.joint_state_subscriber_right = self.node.create_subscription(
            JointState, "/right_arm/joint_states", self._joint_state_callback_right, 10)
        self.eef_pose_subscriber_left = self.node.create_subscription(
            PoseStamped, "/left_arm/eef_pose", self._eef_pose_callback_left, qos_profile_reliable_volatile)
        self.eef_pose_subscriber_right = self.node.create_subscription(
            PoseStamped, "/right_arm/eef_pose", self._eef_pose_callback_right, qos_profile_reliable_volatile)
        self.image_subscriber_left = self.node.create_subscription(
            Image, "/left_cam/image_raw", self._image_callback_left, qos_profile_best_effort_volatile)
        self.image_subscriber_right = self.node.create_subscription(
            Image, "/right_cam/image_raw", self._image_callback_right, qos_profile_best_effort_volatile)

        # Start spinning in a background thread
        self.spin_thread.start()
        self.node.get_logger().info("ROS2 Interface Initialized.")

    # --- Callback Methods ---
    def _joint_state_callback_left(self, msg: JointState):
        with self._lock:
            self._latest_joint_state_left = msg

    def _joint_state_callback_right(self, msg: JointState):
        with self._lock:
            self._latest_joint_state_right = msg

    def _eef_pose_callback_left(self, msg: PoseStamped):
        with self._lock:
            self._latest_eef_pose_left = msg

    def _eef_pose_callback_right(self, msg: PoseStamped):
        with self._lock:
            self._latest_eef_pose_right = msg

    def _image_callback_left(self, msg: Image):
        with self._lock:
            self._latest_image_left = msg

    def _image_callback_right(self, msg: Image):
        with self._lock:
            self._latest_image_right = msg

    # --- Getters for Latest Data ---
    def get_latest_joint_state(self, arm: str) -> Optional[JointState]:
        with self._lock:
            if arm == 'left':
                return self._latest_joint_state_left
            elif arm == 'right':
                return self._latest_joint_state_right
            else:
                raise ValueError("Invalid arm specified")

    def get_latest_eef_pose(self, arm: str) -> Optional[PoseStamped]:
        with self._lock:
            if arm == 'left':
                return self._latest_eef_pose_left
            elif arm == 'right':
                return self._latest_eef_pose_right
            else:
                raise ValueError("Invalid arm specified")

    def get_latest_image(self, arm: str) -> Optional[Image]:
        with self._lock:
            if arm == 'left':
                return self._latest_image_left
            elif arm == 'right':
                return self._latest_image_right
            else:
                raise ValueError("Invalid arm specified")

    # --- Publishers ---
    def publish_eef_command(self, arm: str, pose: np.ndarray, timestamp: Optional[float] = None):
        """ Publishes target EEF pose command. pose is [x,y,z,ax,ay,az] """
        if pose.shape != (6,):
            raise ValueError("Pose must be a 6D numpy array (pos + axis-angle rot_vec)")

        pose_msg = PoseStamped()
        if timestamp is None:
            pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        else:
            sec = int(timestamp)
            nanosec = int((timestamp - sec) * 1e9)
            pose_msg.header.stamp.sec = sec
            pose_msg.header.stamp.nanosec = nanosec

        pose_msg.pose.position = Point(x=pose[0], y=pose[1], z=pose[2])
        rot = st.Rotation.from_rotvec(pose[3:])
        quat_xyzw = rot.as_quat() # SciPy uses xyzw
        pose_msg.pose.orientation = Quaternion(x=quat_xyzw[0], y=quat_xyzw[1], z=quat_xyzw[2], w=quat_xyzw[3]) # ROS2 uses xyzw

        if arm == 'left':
            self.eef_command_publisher_left.publish(pose_msg)
        elif arm == 'right':
            self.eef_command_publisher_right.publish(pose_msg)
        else:
            raise ValueError("Invalid arm specified")

    def publish_gripper_command(self, arm: str, value: float, timestamp: Optional[float] = None):
        """ Publishes target gripper command. """
        msg = Float64()
        msg.data = float(value) # Ensure it's a standard float

        if arm == 'left':
            self.gripper_command_publisher_left.publish(msg)
        elif arm == 'right':
            self.gripper_command_publisher_right.publish(msg)
        else:
            raise ValueError("Invalid arm specified")

    def shutdown(self):
        self.node.get_logger().info("Shutting down ROS2 Interface...")
        self.executor.shutdown()
        self.node.destroy_node()
        # rclpy.shutdown() # Avoid shutting down globally if other nodes exist
# --- END OF umi/real_world/ros2_interface.py ---