import time
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64

class SimGripperNode(Node):
    """Node to subscribe to and publish simulated gripper data."""
    def __init__(self, gripper_id):
        super().__init__(f'sim_gripper_node_{time.time_ns()}')
        self.gripper_id = gripper_id
        arm_prefix = "left_arm" if gripper_id == 0 else "right_arm"

        # Data storage
        self.latest_gripper_state = None

        # Create subscriptions
        grip_topic = f'/{arm_prefix}/gripper_state'
        self.get_logger().info(f"Subscribing to gripper {gripper_id} topic: {grip_topic}")
        self.gripper_state_sub = self.create_subscription(
            Float64,
            grip_topic,
            self.gripper_state_callback,
            10)

        # Create publishers
        self.target_gripper_pub = self.create_publisher(
            Float64,
            f'/{arm_prefix}/target_gripper_width',
            10)

    def gripper_state_callback(self, msg: Float64):
        """Handles incoming gripper state messages."""
        try:
            recv_time = time.time()
            ros_time_sec = recv_time
            gripper_width = np.array([msg.data], dtype=np.float32)

            self.latest_gripper_state = (recv_time, ros_time_sec, gripper_width)
        except Exception as e:
            self.get_logger().error(f"Error processing gripper state msg: {e}", throttle_duration_sec=1.0)

    def get_gripper_state(self):
        """Get gripper state data from the buffer."""
        if self.latest_gripper_state is not None:
            print(f"latest_gripper_state: {self.latest_gripper_state}")
            recv_time, ros_time_sec, gripper_pos = self.latest_gripper_state
            gripper_pos = gripper_pos.astype(np.float32)
            return recv_time, ros_time_sec, gripper_pos
        else:
            # Return default values when no data is received yet
            return time.time(), time.time(), np.array([0.0], dtype=np.float32)

    def send_target_gripper(self, target_width: float):
        """Send target gripper width."""
        msg = Float64()
        msg.data = target_width
        self.target_gripper_pub.publish(msg) 