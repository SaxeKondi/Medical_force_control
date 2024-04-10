from pynput import keyboard
import numpy as np
import time

class KeyboardInput:
    def __init__(self):
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)
        self.gripper_position = 0
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.start = False

    def on_press(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                linear_mapping = {
                    'w': [0.05, 0, 0],
                    's': [-0.05, 0, 0],
                    'a': [0, 0.05, 0],
                    'd': [0, -0.05, 0],
                    'q': [0, 0, 0.05],
                    'e': [0, 0, -0.05],
                }
                self.linear_velocity = np.array(linear_mapping[key_char])
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                angular_mapping = {
                    'i': [0.25, 0, 0],
                    'k': [-0.25, 0, 0],
                    'j': [0, 0.25, 0],
                    'l': [0, -0.25, 0],
                    'u': [0, 0, 0.25],
                    'o': [0, 0, -0.25],
                }
                self.angular_velocity = np.array(angular_mapping[key_char])

        except AttributeError:
            if key == keyboard.Key.space:
                self.start = True

    def on_release(self, key):
        try:
            key_char = key.char.lower()
            if key_char in ['w', 'a', 's', 'd', 'q', 'e']:
                self.linear_velocity = np.zeros(3)
            elif key_char in ['i', 'j', 'k', 'l', 'u', 'o']:
                self.angular_velocity = np.zeros(3)
        except AttributeError:
            pass

    def get_action(self):
        action = np.concatenate((self.linear_velocity, self.angular_velocity))
        return action
    
    def wait_for_start(self, env):
        timeout = 180.0
        start_time = time.time()
        while not self.start:
            if not env.viewer.is_running():
                return False
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print("Timeout while waiting for user to start the simulation.")
                return False
        return True