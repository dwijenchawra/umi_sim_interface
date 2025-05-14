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
        self.latest_camera_data = [None] * self.num_cameras
        self.latest_eef_pose = [None] * self.num_robots
        self.latest_gripper_state = [None] * self.num_robots

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
            

    def is_ready(self):
        """Check if the node is ready."""
        with self.ros_data_lock:
            print("is_ready: ", self.latest_camera_data, self.latest_eef_pose, self.latest_gripper_state)
            return all(data is not None for data in self.latest_camera_data + self.latest_eef_pose + self.latest_gripper_state)

    def _get_ros_time_sec(self, msg) -> float:
        """Extract time in seconds from ROS message header."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def camera_callback(self, msg: Image, camera_idx: int):
        """Handles incoming camera image messages."""
        print(f"camera_callback: {msg.header.stamp.sec} {msg.header.stamp.nanosec}")
        try:
            recv_time = time.time()
            ros_time_sec = self._get_ros_time_sec(msg)
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_transformed = self.image_transforms[camera_idx](img)
            if self.obs_float32 and img_transformed.dtype == np.uint8:
                img_transformed = img_transformed.astype(np.float32) / 255.0
            img_transformed_chw = np.moveaxis(img_transformed, -1, 0)

            with self.ros_data_lock:
                self.latest_camera_data[camera_idx] = (recv_time, ros_time_sec, img_transformed_chw)
        except Exception as e:
            self.get_logger().error(f"Error processing camera {camera_idx} msg: {e}", throttle_duration_sec=1.0)

    def eef_pose_callback(self, msg: PoseStamped, robot_idx: int):
        print(f"eef_pose_callback: {msg.header.stamp.sec} {msg.header.stamp.nanosec}")
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
                self.latest_eef_pose[robot_idx] = (recv_time, ros_time_sec, pose_6d)
        except Exception as e:
            self.get_logger().error(f"Error processing EEF pose {robot_idx} msg: {e}", throttle_duration_sec=1.0)

    def gripper_state_callback(self, msg: Float64, robot_idx: int):
        print(f"gripper_state_callback: {msg.data}")
        """Handles incoming gripper state messages."""
        try:
            recv_time = time.time()
            ros_time_sec = recv_time
            gripper_width = np.array([msg.data], dtype=np.float32)

            with self.ros_data_lock:
                self.latest_gripper_state[robot_idx] = (recv_time, ros_time_sec, gripper_width)
        except Exception as e:
            self.get_logger().error(f"Error processing gripper state {robot_idx} msg: {e}", throttle_duration_sec=1.0)
            
    def get_camera_data(self, camera_topic: str):
        """Get camera data from the buffer."""
        if camera_topic == '/left_arm/camera':
            camera_idx = 0
        elif camera_topic == '/right_arm/camera':
            camera_idx = 1
        
        with self.ros_data_lock:
            return self.latest_camera_data[camera_idx]
    
    def get_eef_pose(self, robot_idx: int) -> tuple[float, float, np.ndarray]:
        """Get end-effector pose data from the buffer.
        """
        with self.ros_data_lock:
            if self.latest_eef_pose[robot_idx] is not None:
                recv_time, ros_time_sec, pose_6d = self.latest_eef_pose[robot_idx]
                pose_6d = pose_6d.astype(np.float32)
                return recv_time, ros_time_sec, pose_6d
            else:
                return None, None, None
            
    def get_gripper_state(self, gripper_idx: int) -> tuple[float, float, np.ndarray]:
        """Get gripper state data from the buffer."""
        with self.ros_data_lock:
            if self.latest_gripper_state[gripper_idx] is not None:
                recv_time, ros_time_sec, gripper_width = self.latest_gripper_state[gripper_idx]
                return recv_time, ros_time_sec, gripper_width
            else:
                return None, None, None
    
    def send_target_pose(self, robot_idx: int, target_pose: np.ndarray):
        # print("sending target pose: ", target_pose)
        """Send target pose to the robot.
        target pose is a 6D vector [x, y, z, rx, ry, rz]
        need to convert to quaternion [x, y, z, w]
        """
        # Iterate through robots
        # Extract pose and gripper action for this robot
        pose_6d = target_pose

        # Create PoseStamped message
        pose_msg = PoseStamped()
        
        now_ros_time = self.get_clock().now()
        target_ros_time = now_ros_time 
        
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

        # print("SENDING TO ROBOT pose_msg: ", pose_msg)
        self.target_pose_pubs[robot_idx].publish(pose_msg)


    
    def send_target_gripper(self, robot_idx: int, target_width: float):
        msg = Float64()
        msg.data = target_width
        self.target_gripper_pubs[robot_idx].publish(msg)

    def send_data_to_stdout(self):
        """Send camera, EEF pose, and gripper data to stdout."""
        import json
        with self.ros_data_lock:
            data = {
                "camera_data": [
                    {
                        "recv_time": cam[0],
                        "ros_time_sec": cam[1],
                        "image": cam[2].tolist() if cam else None
                    } for cam in self.latest_camera_data
                ],
                "eef_pose": [
                    {
                        "recv_time": pose[0],
                        "ros_time_sec": pose[1],
                        "pose_6d": pose[2].tolist() if pose else None
                    } for pose in self.latest_eef_pose
                ],
                "gripper_state": [
                    {
                        "recv_time": grip[0],
                        "ros_time_sec": grip[1],
                        "gripper_width": grip[2].tolist() if grip else None
                    } for grip in self.latest_gripper_state
                ]
            }
            print(json.dumps(data))

    def main_loop(self):
        """Main loop to send data to stdout."""
        while rclpy.ok():
            rclpy.spin_once(self)
            self.send_data_to_stdout()
            time.sleep(0.01)  # Add a small delay to prevent busy-waiting

# Replace the main function
if __name__ == '__main__':
    rclpy.init()
    node = RosSimInterfaceNode(num_robots=2, num_cameras=2, obs_float32=False, image_transforms=None)
    try:
        node.main_loop()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()