import numpy as np
import random
import gym
import os
from datetime import datetime
from gym import spaces
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Custom Environment that follows gym interface
class UnicycleEnv(gym.Env):
    matadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(UnicycleEnv, self).__init__()
        self.dynamiuc_mode = 'Unicycle'
        
        self.action_space = spaces.Box(low=-4, high=4, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1e10, high=1e10, shape=(7,), dtype=np.float32)
        
        self.dt = 0.02
        self.max_episode_step = 5000
        # self.reward_goal = 1.0
        
        self.goal_pos = None
        self.episode_step = 0
        
        self.render_flag = False
        
        self.reset()
        
        self.get_f, self.get_g = self._get_dynamics()
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20
  
    def reset(self, rand_init=True):
        self.episode_step = 0
        rand = rand_init
        # 10 secind per circle is the base speed
        if rand:
            # self.circle_r = random.uniform(1, 3)
            self.circle_r = 2
            self.speed = 2 / self.circle_r
            self.initial_angle = random.uniform(0, 2 * np.pi)
            self.state = np.array([self.circle_r * np.cos(self.initial_angle), self.circle_r * np.sin(self.initial_angle), random.uniform(0, 2 * np.pi)])
            self.goal_pos = [self.circle_r * np.cos(self.initial_angle), self.circle_r * np.sin(self.initial_angle)]
        else:
            self.goal_pos = [2, 0]
            self.state = np.array([2., 0., np.pi/2.])
            self.initial_angle = 0
            self.speed = 1
            self.circle_r = 2
        # self.last_goal_dist = self._goal_dist()
        
        if self.render_flag:
            self.render_start()
        
        return self.get_obs()

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Update the state
        # With disturbance
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * -0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0]) * np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()
        self.episode_step += 1
        
        # Get info to satisfy the gym interface
        info = dict()
        
        # Get the reward 
        reward = - self._goal_dist()
        # if self._goal_dist() < 0.05:
        #     reward += self.reward_goal
        # self.last_goal_dist = dist_goal
        self.goal_pos = [self.circle_r * np.cos(self.speed * np.pi/5 * self.episode_step * self.dt + self.initial_angle), self.circle_r * np.sin(self.speed * np.pi/5 * self.episode_step * self.dt + self.initial_angle)]
        
        # Check if the episode is done
        done = self.episode_step >= 2000
        
        return self.get_obs(), reward, done, info  
    
    # state = [x in world frame, y in world frame, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
    def get_obs(self):
        goal_dist = self._goal_dist()
        goal_compass = self.state_compass()
    
        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])  
    
    # Get the normalized vector toward the goal position in the robot frame
    def state_compass(self):
        vec = self.goal_pos - self.state[:2]
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec
    
    def render_start(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(-4, 4)
        self.ax.set_ylim(-4, 4)
        self.ax.set_aspect('equal')
        
        circle_inner = plt.Circle((0, 0), self.circle_r, color='black', fill=False, linestyle='--',linewidth=1)
        self.ax.add_artist(circle_inner)
        
        self.path_data, = self.ax.plot([], [], 'g--', linewidth=1)
        self.robot, = self.ax.plot([], [], 'ro', markersize=5)
        self.target, = self.ax.plot([], [], 'bo', markersize=5)
        self.x_robot = []
        self.y_robot = []
        self.x_target = []
        self.y_target = []
        
        self.save_folder = "animations"
        os.makedirs(self.save_folder, exist_ok=True)
    
    def render_save(self):
        self.x_robot.append(self.state[0])
        self.y_robot.append(self.state[1])
        self.x_target.append(self.goal_pos[0])
        self.y_target.append(self.goal_pos[1])
        
    def init(self):
        self.path_data.set_data([], [])
        self.robot.set_data([], [])
        self.target.set_data([], [])
        return self.path_data, self.robot, self.target
        
    def update(self, frame):
        i = int(frame)
        self.robot.set_data(self.x_robot[i], self.y_robot[i])
        self.target.set_data(self.x_target[i], self.y_target[i])
        old_x, old_y = self.path_data.get_data()
        new_x = np.append(old_x, self.x_robot[i])
        new_y = np.append(old_y, self.y_robot[i])
        self.path_data.set_data(new_x, new_y)
        
        return self.path_data, self.robot, self.target

    def render_activate(self):  
        print("Rendering")     
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = f"{self.save_folder}/Unicycle_{timestamp}.mp4"
        
        ani = FuncAnimation(self.fig, self.update, frames=np.linspace(0, len(self.x_robot)-1, len(self.x_robot)),init_func=self.init, blit=True, interval=1000/24)
        ani.save(filename, fps=24, extra_args=['-vcodec', 'libx264'])
        plt.close(self.fig)  
        plt.close('all')  
           
        
    # Get the distance between the current position and the goal position
    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2] )
    
    # The control is based in the current robot frame
    def _get_dynamics(self):
        # Drift dynamics of the continuous syste, x' = f(x) + g(x)u
        def get_f(state):
            f_x = np.zeros(state.shape)
            return f_x

        # Control dynamics of the continuous system, x' = f(x) + g(x)u
        def get_g(state):
            theta = state[2]
            g_x = np.array([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [            0, 1.0]])
            return g_x
        return get_f, get_g