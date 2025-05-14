import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait

from umi.sim.sim_arm_node import SimArmNode
import rclpy
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2


class SimURController(mp.Process):
    """
    To ensure sending command to the robot with predictable latency
    this controller need its separate process (due to python GIL)
    """


    def __init__(self,
            shm_manager: SharedMemoryManager,
            robot_id,
            frequency=125, 
            lookahead_time=0.1, 
            soft_real_time=False,
            verbose=False,
            receive_keys=None,
            get_max_k=None,
            receive_latency=0.0
            ):
        """
        frequency: CB2=125, UR3e=500
        lookahead_time: [0.03, 0.2]s smoothens the trajectory with this lookahead time
        gain: [100, 2000] proportional gain for following target position
        max_pos_speed: m/s
        max_rot_speed: rad/s
        tcp_offset_pose: 6d pose
        payload_mass: float
        payload_cog: 3d position, center of gravity
        soft_real_time: enables round-robin scheduling and real-time priority
            requires running scripts/rtprio_setup.sh before hand.

        """
        # verify
        assert 0 < frequency <= 500
        assert 0.03 <= lookahead_time <= 0.2
        # if tcp_offset_pose is not None:
        #     tcp_offset_pose = np.array(tcp_offset_pose)
        #     assert tcp_offset_pose.shape == (6,)
        # if joints_init is not None:
        #     joints_init = np.array(joints_init)
        #     assert joints_init.shape == (6,)

        super().__init__(name="SimURController")
        self.robot_id = robot_id
        self.frequency = frequency
        self.lookahead_time = lookahead_time
        # self.tcp_offset_pose = tcp_offset_pose
        # self.joints_init = joints_init
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose
        self.launch_timeout = 3

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # build input queue
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # Build ring buffer for ROS-based robot state data
        example = {
            'eef_pose': np.zeros((6,), dtype=np.float32),  # [x, y, z, rx, ry, rz]
            'ros_time': 0.0,
            'recv_time': 0.0
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
        
    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[RTDEPositionalController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
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
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
            import traceback
            traceback.print_tb(exc_tb)

        self.stop()
        
    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        """
        duration: desired time to reach pose
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    # def get_state(self, k=None, out=None):
    #     if k is None:
    #         return self.ring_buffer.get(out=out)
    #     else:
    #         return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # Create arm node
        self.arm_node = SimArmNode(robot_id=self.robot_id)
        
        try:
            if self.verbose:
                print(f"[RTDEPositionalController] Connect to robot")

            # main loop
            dt = 1. / self.frequency
            recv_time, ros_time, curr_pose = self.arm_node.get_eef_pose()
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )
            
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # Spin ROS node once
                rclpy.spin_once(self.arm_node)
                
                t_now = time.monotonic()
                pose_command = pose_interp(t_now) # returns 6d tensor
                self.arm_node.send_target_pose(
                    # robot_idx=self.robot_id,
                    target_pose=pose_command
                )
                
                # update robot state
                recv_time, ros_time, eef_pose = self.arm_node.get_eef_pose()
                t_recv = time.time()
                state = {
                    'eef_pose': eef_pose,
                    'ros_time': ros_time,
                    'recv_time': t_recv
                }
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                    # process at most 1 command per cycle to maintain frequency
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                            max_pos_speed=100,
                            max_rot_speed=100
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[RTDEPositionalController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=100,
                            max_rot_speed=100,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1
                
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                if self.verbose:
                    print(f"[RTDEPositionalController] Actual frequency {1/(time.monotonic() - t_now)}")

        finally:
            self.arm_node.destroy_node()
            self.ready_event.set()

            if self.verbose:
                print(f"[RTDEPositionalController] Disconnected from robot")
