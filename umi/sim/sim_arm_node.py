import time
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import scipy.spatial.transform as st

class SimArmNode(Node):
    """Node to subscribe to and publish simulated robot data."""
    def __init__(self, robot_id):
        super().__init__(f'sim_arm_node_{time.time_ns()}')
        self.robot_id = robot_id
        arm_prefix = "left_arm" if robot_id == 0 else "right_arm"

        # Data storage
        self.latest_eef_pose = None

        # Create subscriptions
        pose_topic = f'/{arm_prefix}/eef_pose_state'
        self.get_logger().info(f"Subscribing to robot {robot_id} pose topic: {pose_topic}")
        self.eef_pose_sub = self.create_subscription(
            PoseStamped,
            pose_topic,
            self.eef_pose_callback,
            10)

        # Create publishers
        self.target_pose_pub = self.create_publisher(
            PoseStamped, 
            f'/{arm_prefix}/target_eef_pose', 
            10)

    def _get_ros_time_sec(self, msg) -> float:
        """Extract time in seconds from ROS message header."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def eef_pose_callback(self, msg: PoseStamped):
        """Handles incoming end-effector pose messages."""
        try:
            recv_time = time.time()
            ros_time_sec = self._get_ros_time_sec(msg)
            position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
            quat_xyzw = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])
            rotation = st.Rotation.from_quat(quat_xyzw)
            rotvec = rotation.as_rotvec()
            pose_6d = np.concatenate([position, rotvec]).astype(np.float32)

            self.latest_eef_pose = (recv_time, ros_time_sec, pose_6d)
        except Exception as e:
            self.get_logger().error(f"Error processing EEF pose msg: {e}", throttle_duration_sec=1.0)

    def get_eef_pose(self):
        """Get end-effector pose data from the buffer."""
        if self.latest_eef_pose is not None:
            print(f"latest_eef_pose: {self.latest_eef_pose}")
            recv_time, ros_time_sec, pose_6d = self.latest_eef_pose
            pose_6d = pose_6d.astype(np.float32)
            return recv_time, ros_time_sec, pose_6d
        else:
            # Return default values when no data is received yet
            return time.time(), time.time(), np.zeros(6, dtype=np.float32)

    def send_target_pose(self, target_pose: np.ndarray):
        """Send target pose to the robot.
        target pose is a 6D vector [x, y, z, rx, ry, rz]
        """
        pose_6d = target_pose

        # Create PoseStamped message
        pose_msg = PoseStamped()
        
        now_ros_time = self.get_clock().now()
        target_ros_time = now_ros_time 
        
        pose_msg.header.stamp = target_ros_time.to_msg()
        pose_msg.header.frame_id = "world"
        pose_msg.pose.position.x = float(pose_6d[0])
        pose_msg.pose.position.y = float(pose_6d[1])
        pose_msg.pose.position.z = float(pose_6d[2])
        
        # Convert rotvec to quaternion
        quat_xyzw = st.Rotation.from_rotvec(pose_6d[3:]).as_quat()
        pose_msg.pose.orientation.x = float(quat_xyzw[0])
        pose_msg.pose.orientation.y = float(quat_xyzw[1])
        pose_msg.pose.orientation.z = float(quat_xyzw[2])
        pose_msg.pose.orientation.w = float(quat_xyzw[3])

        self.target_pose_pub.publish(pose_msg) 