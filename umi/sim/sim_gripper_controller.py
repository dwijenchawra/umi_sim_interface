import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.real_world.wsg_binary_driver import WSGBinaryDriver
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from umi.sim.ros_interface import RosSimInterfaceNode

class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class SimGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            ros_node: RosSimInterfaceNode,
            frequency=10,
            command_queue_size=1024,
            verbose=False
            ):
        super().__init__(name="SimGripperController")
        self.verbose = verbose
        self.ros_node = ros_node
        self.launch_timeout = 3
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_position': 0.0,
            'ros_time': 0.0,
            'recv_time': 0.0
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=32,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.frequency = frequency
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[WSGController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        try:
            keep_running = True
            while keep_running:
                t_now = time.monotonic()

                # Fetch gripper state from ROS
                recv_time, ros_time, gripper_width = self.ros_node.get_gripper_state(robot_idx=0)
                assert gripper_width is not None, "Gripper width is None"
                assert ros_time is not None, "ROS time is None"
                assert recv_time is not None, "Receive time is None"
                state = {
                    'gripper_position': gripper_width[0],
                    'ros_time': ros_time,
                    'recv_time': recv_time
                }
                self.ring_buffer.put(state)

                # Fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        self.ros_node.send_target_gripper(robot_idx=0, target_width=target_pos)

                # Regulate frequency
                time.sleep(1 / self.frequency)

        finally:
            self.ready_event.set()
            if self.verbose:
                print("[SimGripperController] Process terminated.")