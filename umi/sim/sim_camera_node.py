import time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SimCameraNode(Node):
    """Node to subscribe to simulated camera data."""
    def __init__(self, camera_topic, obs_float32=False, image_transform=None):
        super().__init__(f'sim_camera_node_{time.time_ns()}')
        self.cv_bridge = CvBridge()
        self.obs_float32 = obs_float32
        self.image_transform = image_transform or (lambda x: x)
        self.camera_topic = camera_topic

        # Data storage
        self.latest_camera_data = None

        # Create subscription
        self.get_logger().info(f"Subscribing to camera topic: {camera_topic}")
        self.camera_sub = self.create_subscription(
            Image,
            camera_topic,
            self.camera_callback,
            10)  # QoS profile depth 10

    def _get_ros_time_sec(self, msg) -> float:
        """Extract time in seconds from ROS message header."""
        return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

    def camera_callback(self, msg):
        """Process incoming camera messages."""
        try:
            # Convert ROS Image message to numpy array
            img = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Convert to CHW format
            img_chw = np.transpose(img, (2, 0, 1))
            
            # Get timestamps
            recv_time = time.time()
            ros_time_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            
            # Store data
            self.latest_camera_data = (recv_time, ros_time_sec, img_chw)
                
        except Exception as e:
            self.get_logger().error(f'Error processing camera msg: {str(e)}')

    def get_camera_data(self):
        """Get latest camera data."""
        return self.latest_camera_data 