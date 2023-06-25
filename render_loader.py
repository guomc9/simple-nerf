import json
from tools import get_rotation_matrix, get_translation_matrix, get_rays_np
import numpy as np
import cv2

class renderLoader:
    def __init__(self, render_json):
        # Load the JSON file
        with open(render_json, "r") as f:
            render_data = json.load(f)

        self.tasks = render_data["tasks"]

    def get_rays(self):
        """get rays origin and directory

        Returns:
            torch.Tensor: rays origin, [N_images*H*W, 3]
            torch.Tensor: rays directory, [N_images*H*W, 3]
            float: near plane Z-axis value
            float: far plane Z-axis value
        """
        task = self.tasks[0]
        self.tasks.pop(0)
        near, far = float(task['near']), float(task['far'])
        H, W, fovY, eye_pos = task['height'], task['width'], float(task['fovY']), np.asarray([task['eye_pos']['x'], task['eye_pos']['y'], task['eye_pos']['z']],dtype=np.float32)
        focal = float(.5 * H / np.tan(fovY / 2))
        view_z_dir = float(task['view_z_dir'])
        self.H, self.W = H, W
        self.task_type = task['task_type']
        if self.task_type == 'V':
            rot_angle, rot_speed = task['rot_angle'], task['rot_speed']
            self.fps, self.video_save_path = task['fps'], task['video_save_path']
            if task['rot_axis']['axis'] == 'y':
                axis = np.asarray([0., 1., 0.])
                axis_coords = np.asarray([task['rot_axis']['coord_1'], 0., task['rot_axis']['coord_2']])
            elif task['rot_axis']['axis'] == 'x':
                axis = np.asarray([1., 0., 0.])
                axis_coords = np.asarray([0., task['rot_axis']['coord_1'], task['rot_axis']['coord_2']])
            c2ws = []
            K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                ], dtype=np.float32)
            self.N_images = 0
            forward = get_translation_matrix(-axis_coords)
            back = get_translation_matrix(axis_coords)
            cur_angle = 0.
            while cur_angle < rot_angle:
                rot = get_rotation_matrix(axis, cur_angle)
                c2w = back @ rot @ forward
                print(c2w)
                c2ws.append(c2w)
                c2ws[-1][:-1,-1] = eye_pos
                cur_angle += rot_speed
                self.N_images += 1
            rays_o, rays_d = get_rays_np(H, W, K, view_z_dir, np.stack(c2ws, axis=0)[:,:-1,:])

        elif self.task_type == 'I':
            self.image_save_path = task["image_save_path"]
            c2w = [np.eye(N=4)]
            c2w[-1][:-1,:-1] = np.asarray(task["pose"])
            c2w[-1][:-1,-1] = eye_pos
            K = np.array([
                    [focal, 0, 0.5*W],
                    [0, focal, 0.5*H],
                    [0, 0, 1]
                ])
            rays_o, rays_d = get_rays_np(H, W, K, view_z_dir, np.stack(c2w, axis=0)[:,:-1,:])
        rays_o = rays_o.reshape(-1, 3).astype(np.float32)
        rays_d = rays_d.reshape(-1, 3).astype(np.float32)
        return rays_o, rays_d, near, far 
        
    def empty(self):
        """tasks is empty

        Returns:
            bool: tasks is empty or not
        """
        return len(self.tasks) == 0
    
    def submit(self, rgb):
        """submit rays rgb

        Args:
            rgb (numpy.ndarray): rays rgb, [N_images*H*W, 3]
        """
        if self.task_type == 'V':
            rgb = np.reshape(rgb, newshape=[self.N_images, self.H, self.W, 3])
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.video_save_path, fourcc, self.fps, (self.W, self.H))
            for i in range(self.N_images):
                frame = rgb[i]
                frame = 255 * np.clip(frame, a_min=0., a_max=1.)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_writer.write(frame)
            video_writer.release()
        elif self.task_type == 'I':
            rgb = np.reshape(rgb, newshape=[self.H, self.W, 3])
            frame = 255 * np.clip(rgb, a_min=0., a_max=1.)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(self.image_save_path, frame)
