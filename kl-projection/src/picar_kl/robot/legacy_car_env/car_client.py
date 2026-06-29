import requests 
from requests.exceptions import Timeout
import json 
import numpy as np 
import cv2
import random 
import pickle 
import os 
from time import time, sleep 
from .constants import BALL_SIZE_MIN, BALL_SIZE_MAX, N_ACTIONS   

## CLIENT 

def img(host, x_resize=None, y_resize=None): 
    params = None 
    if x_resize is not None and y_resize is not None: 
        params = {'x_resize': x_resize, 'y_resize': y_resize} 
        pass 
    r = __handle_get_response(f'http://{host}/img', params=params)
    content_type = r.headers.get('Content-Type', '')

    if 'application/json' in content_type:
        image, x, y, r_blob = json.loads(r.text)
        return np.array(image).astype(np.uint8), x, y, r_blob

    image_bytes = np.frombuffer(r.content, dtype=np.uint8)
    decoded_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise Exception('ERROR! Failed to decode image bytes from `/img` response.')
    decoded_rgb = cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)

    # Blob metadata is no longer returned by `/img`; keep interface stable.
    x = int(r.headers.get('X-Ball-X', 0))
    y = int(r.headers.get('X-Ball-Y', 0))
    r_blob = int(r.headers.get('X-Ball-R', 0))
    return decoded_rgb.astype(np.uint8), x, y, r_blob  

def drive_left(host): 
    __handle_get(f'http://{host}/drive-left') 
    pass 

def drive_right(host): 
    __handle_get(f'http://{host}/drive-right') 
    pass 

def drive_forward(host): 
    __handle_get(f'http://{host}/drive-forward') 
    pass 

def drive_backward(host): 
    __handle_get(f'http://{host}/drive-backward') 
    pass 

def look_left(host): 
    __handle_get(f'http://{host}/look-left') 
    pass 

def look_right(host): 
    __handle_get(f'http://{host}/look-right') 
    pass 

def look_up(host): 
    __handle_get(f'http://{host}/look-up') 
    pass 

def look_forward(host): 
    __handle_get(f'http://{host}/look-forward') 
    pass 

def apply_vector(host, pan=0.0, tilt=0.0, turn=0.0, drive=0.0):
    params = {
            'pan': pan,
            'tilt': tilt,
            'turn': turn,
            'drive': drive,
            }
    t = __handle_get(f'http://{host}/apply_vector', params=params)
    return json.loads(t)

def __handle_get(url, params=None): 
    r = __handle_get_response(url=url, params=params)
    return r.text 

def __handle_get_response(url, params=None):
    retries = 5 ## important because raspberrypi servers respond intermittently under load 
    continue_attempting = True 
    while continue_attempting:
        try: 
            r = requests.get(url, params=params, timeout=10) 
            continue_attempting = False 
        except Timeout as e: 
            print(f'WARNING! timeout occurred! Error message:\n{e}') 
            retries -= 1 
            sleep(.1) 
            if retries <= 0: 
                raise Exception('ERROR: Out of timeouts!') 
                pass 
            pass 
        retries = 0 
    if r.status_code != 200: 
        raise Exception(f'ERROR! Status code: {r.status_code}') 
    return r

## RL ENV 

class PiCarEnv(): 
    def __init__(self, host, cool_down=.2, memory_length=0, memory_write_location='/tmp', x_resize=None, y_resize=None): 
        self.host = host 
        self.cool_down = cool_down 
        self.memory_length = memory_length ## <= 0 disables memorization 
        self.memory_write_location = memory_write_location ## where to write memory when full 
        look_forward(self.host) 
        self.last_action = 7 
        self.camera = 0 
        self.x_resize = x_resize 
        self.y_resize = y_resize 
        self.get_image() 
        self.last_action_time = time() 
        self.memory = [] 
        pass 
    def reset(self): 
        look_forward(self.host) 
        self.camera = 0 
        self.get_image() 
        return self.last_image, self.camera  
    def action_space_sample(self): 
        return random.randint(0, N_ACTIONS-1) ## sampling range is inclusive 
    def step(self, action): 
        if action in PiCarEnv.__action_dict: 
            PiCarEnv.__action_dict[action](self.host) 
            pass ## implicit else: no-op 
        if action in PiCarEnv.__look_dict: 
            self.camera = PiCarEnv.__look_dict[action] 
            pass 
        self.last_action = action 
        self.get_image() 
        self.last_action_time = time() 
        self.memorize() 
        reward = PiCarEnv.get_reward(self.last_r) 
        return self.last_image, self.camera, reward  
    def auto_action(self): 
        t = time()
        if t - self.last_action_time < self.cool_down: 
            ## avoid overheats 
            sleep(self.cool_down - (t - self.last_action_time)) 
            pass 
        action = self.suggest_action() 
        self.step(action) 
        return self.last_image, self.last_x, self.last_y, self.last_r 
    def get_image(self): 
        self.last_image, self.last_x, self.last_y, self.last_r = img(self.host, x_resize=self.x_resize, y_resize=self.y_resize) 
        pass 
    def suggest_action(self): 
        ## vision constants 
        left = 20 
        center = 30 
        right = 40 
        ## name inputs 
        ball_x_location = self.last_x 
        ball_radius = self.last_r 
        if ball_radius == 0: 
            ## no ball seen 
            ## scan for it 
            if self.camera == 0:
                return 4 ## look_right 
            elif self.camera == 1: 
                return 6 ## look_up 
            elif self.camera == 3: 
                return 5 ## look_right 
            else: 
                return 7 ## look_forward 
        elif self.camera == 1: 
            ## looking left 
            if ball_x_location < center: 
                ## drive left, toward it 
                return 0 ## drive_left 
            else: 
                ## center the camera 
                return 7 ## look_forward 
        elif self.camera == 2: 
            ## looking right 
            if ball_x_location > center: 
                ## drive right, toward it 
                return 1 ## drive_right 
            else: 
                ## center the camera 
                return 7 ## look foward 
        else: 
            ## camera is foward or up 
            if ball_x_location < left: 
                ## drive left, toward it 
                return 0 ## drive_left 
            elif ball_x_location > right: 
                return 1 ## drive_right 
            else: 
                ## ball is center 
                if ball_radius < BALL_SIZE_MAX: 
                    return 2 ## drive_forward 
                else: 
                    return 3 ## drive_backward 
        pass 
    @staticmethod 
    def get_reward(r): 
        ball_radius = r 
        ## reward being close to the ball 
        if ball_radius >= BALL_SIZE_MIN and ball_radius <= BALL_SIZE_MAX: 
            return(ball_radius - BALL_SIZE_MIN ) / (BALL_SIZE_MAX - BALL_SIZE_MIN) 
        elif ball_radius > BALL_SIZE_MAX: 
            ## slight punishment for being too close 
            return min(1. + (BALL_SIZE_MAX - ball_radius)/20., -.1) 
            pass 
        return 0. 
    def memorize(self): 
        if self.memory_length <= 0: 
            ## memorization disabled 
            return None 
        memory_tuple = ( 
                self.last_image, 
                self.last_action,
                self.last_x, 
                self.last_r, 
                self.camera 
                ) 
        self.memory.append(memory_tuple) 
        if len(self.memory) > self.memory_length: 
            ## memory full, write to disk 
            ## first, convert to lists 
            image_list = [] 
            action_list = [] 
            x_list = [] 
            r_list = [] 
            camera_list = [] 
            for memory_tuple in self.memory: 
                image_list.append(memory_tuple[0]) 
                action_list.append(memory_tuple[1]) 
                x_list.append(memory_tuple[2]) 
                r_list.append(memory_tuple[3]) 
                camera_list.append(memory_tuple[4]) 
                pass 
            ## stack images for space efficiency 
            image_list = np.stack(image_list) 
            ## pickle 
            data = (image_list, action_list, x_list, r_list) 
            if self.memory_write_location is not None: 
                filename = os.path.join(self.memory_write_location, 'picar-memory-'+str(int(time()))+'.pkl') 
                with open(filename, 'wb') as f: 
                    pickle.dump(data, f) 
                    pass 
                pass 
            ## forget 
            self.memory = [] 
        pass 
    @staticmethod 
    def load_memory(filepath): 
        with open(filepath, 'rb') as f: 
            data = pickle.load(f) 
            pass 
        unstacked_arrays = PiCarEnv.__unstack(data[0]) 
        unstacked_data = (
                unstacked_arrays, ## images 
                data[1], ## actions 
                data[2], ## ball x locations 
                data[3], ## ball radii 
                data[4]  ## camera positions 
                ) 
        return unstacked_data 

    @staticmethod 
    def __unstack(a, axis = 0):
        return [np.squeeze(e, axis) for e in np.split(a, a.shape[axis], axis = axis)]

    __action_dict = { 
            0: drive_left, 
            1: drive_right, 
            2: drive_forward, 
            3: drive_backward, 
            4: look_left, 
            5: look_right, 
            6: look_up, 
            7: look_forward 
            } 
    __look_dict = { ## map action ints to camera state ints 
            4: 1, ## left 
            5: 2, ## right 
            6: 3, ## up 
            7: 0  ## foward 
            } 
    pass 
