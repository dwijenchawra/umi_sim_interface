import time
import multiprocessing as mp
import numpy as np
import cv2
from threadpoolctl import threadpool_limits

class MultiCameraVisualizer(mp.Process):
    def __init__(self,
        camera,
        row, col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True
        ):
        super().__init__()
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr=rgb_to_bgr
        self.camera = camera
        # shared variables
        self.stop_event = mp.Event()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):
        cv2.setNumThreads(1)
        threadpool_limits(1)
        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None,None,-1)

        vis_data = None
        vis_img = None
        while not self.stop_event.is_set():
            vis_data = self.camera.get_vis(out=vis_data)
            color = vis_data['color']
            N, H, W, C = color.shape
            assert C == 3
            oh = H * self.row
            ow = W * self.col
            if vis_img is None:
                vis_img = np.full((oh, ow, 3), 
                    fill_value=self.fill_value, dtype=np.uint8)
            for row in range(self.row):
                for col in range(self.col):
                    idx = col + row * self.col
                    h_start = H * row
                    h_end = h_start + H
                    w_start = W * col
                    w_end = w_start + W
                    if idx < N:
                        # opencv uses bgr
                        vis_img[h_start:h_end,w_start:w_end
                            ] = color[idx,:,:,channel_slice]
            cv2.imshow(self.window_name, vis_img)
            cv2.pollKey()
            time.sleep(1 / self.vis_fps)


class SimMultiCameraVisualizer(mp.Process):
    def __init__(self,
        env_instance,
        row, col,
        window_name='Multi Cam Vis',
        vis_fps=60,
        fill_value=0,
        rgb_to_bgr=True
        ):
        super().__init__()
        self.env_instance = env_instance  # Reference to the simulation environment
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr = rgb_to_bgr
        self.stop_event = mp.Event()

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self, wait=False):
        super().start()
    
    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()        
    
    def run(self):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        threadpool_limits(1)
        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None, None, -1)

        fig, ax = plt.subplots()
        img_display = ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))  # Placeholder for initialization
        plt.ion()
        plt.show()

        while not self.stop_event.is_set():
            with self.env_instance.ros_data_lock:
                # Fetch the latest images from the camera buffers
                camera_buffers = self.env_instance.camera_buffers
                latest_image = None
                for buffer in camera_buffers:
                    if len(buffer) > 0:
                        # Get the latest image (index 2 in the buffer tuple)
                        latest_image = buffer[-1][2]
                        break

                if latest_image is None:
                    # If no image is available, use a placeholder
                    latest_image = np.full(
                        (self.env_instance.obs_image_resolution[1], 
                         self.env_instance.obs_image_resolution[0], 3),
                        fill_value=self.fill_value, dtype=np.uint8)

            # Convert image from CHW to HWC
            latest_image = np.transpose(latest_image, (1, 2, 0))
            if self.rgb_to_bgr:
                latest_image = latest_image[..., channel_slice]

            # Update the matplotlib display
            img_display.set_data(latest_image)
            ax.set_title(self.window_name)
            plt.pause(1 / self.vis_fps)
