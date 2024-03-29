import numpy as np
import random
import gym
from gym import spaces
from envs.utils import to_pixel

# Custom Environment that follows gym interface
class UnicycleEnv(gym.Env):
    matadata = {'render.modes': ['human']}
    
    def __init__(self, obs_config = 'default', rand_init=True):
        super(UnicycleEnv, self).__init__()
        self.dynamic_mode = 'Unicycle'
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low = -1e10, high = 1e10, shape=(7,), dtype=np.float32)
        # self.bds = np.array([[-3., -3.], [3., 3.]])
        
        # Define the dynamics interpretation of the simulator
        self.dt = 0.02
        self.max_episode_steps = 5000
        
        # Define the state and the goal
        # The target movement is to run a counterclockwise circle
        self.goal_pos = None
        self.episode_step = 0
        self.initial_state = np.array([0., 0., 0.,])
        self.state = np.copy(self.initial_state)
        self.rand_init = rand_init
        
        # Generate the initual state
        self.reset()

        # Get dynamics of the uni-cycle
        # These are two functions
        self.get_f, self.get_g = self._get_dynamics()
        #Disturbance parameters
        self.disturb_mean = np.zeros((3,))
        self.disturb_covar = np.diag([0.005, 0.005, 0.05]) * 20

        # For rendering
        self.viewer = None
      
    # Reset the state of the environment to an initial state
    # Return the initial state
    def reset(self):
        self.episode_step = 0
        # Generate the initial state
        # 10 second per circle is the base speed
        if self.rand_init:
            self.circle_r = random.uniform(1, 2.5)
            self.speed = 1 * random.uniform(1, 2.5)
            circle_angle = random.uniform(0, 2 * np.pi)
            self.goal_pos = [self.circle_r * np.cos(circle_angle),self.circle_r * np.sin(circle_angle)]
        else:
            self.goal_pos = [1, 0]
            self.speed = 1
        self.last_goal_dist = self._goal_dist()
        
        return self.get_obs()
       
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        state, reward, done, info = self._step(action)
        return self.get_obs(), reward, done, info        
        
    # Return the current state
    # state = [x in world frame, y in world frame, cos(theta), sin(theta), xdir2goal, ydir2goal, exp(-dist2goal)]
    def get_obs(self):
        goal_dist = self._goal_dist()
        goal_compass = self.state_compass()
    
        return np.array([self.state[0], self.state[1], np.cos(self.state[2]), np.sin(self.state[2]), goal_compass[0], goal_compass[1], np.exp(-goal_dist)])    
    
    # Get the normalized vector toward the goal position in the robot frame
    def state_compass(self):
        # Get the direction vector in world frame
        vec = self.goal_pos - self.state[:2]
        # Rotate to robot frame
        R = np.array([[np.cos(self.state[2]), -np.sin(self.state[2])], [np.sin(self.state[2]), np.cos(self.state[2])]])
        vec = np.matmul(vec, R)
        # Normalize the vector
        vec /= np.sqrt(np.sum(np.square(vec))) + 0.001
        return vec
    
    # render the situation
    def render(self, mode='human', close = False):
        # Just print the state
        if mode != 'human' and mode != 'rgb_array':
            rel_loc = self.goal_pos - self.state[:2]
            theta_error = np.arctan2(rel_loc[1], rel_loc[0]) - self.state[2]
            print('Ep_step = {}, \tState = {}, \tDist2Goal = {}, alignment_error = {}'.format(self.episode_step, self.state, self._goal_dist(), theta_error))
            
        screen_width = 600
        screen_height = 400
        
        if self.viewer is None:
            from envs import pyglet_rendering
            self.viewer = pyglet_rendering.Viewer(screen_width, screen_height)
            
            # Make goal
            goal = pyglet_rendering.make_circle(radius=to_pixel(0.1, shift=0), filled=True)
            goal_trans = pyglet_rendering.Transform(translation=(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2)))
            goal.add_attr(goal_trans)
            goal.set_color(0.0, 0.5, 0.0)
            self.viewer.add_geom(goal)
            
            #Make robot
            self.robot = pyglet_rendering.make_circle(radius=to_pixel(0.1), filled=True)
            self.robot_trans = pyglet_rendering.Transform(translation=(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2)))
            self.robot_trans.set_rotation(self.state[2])
            self.robot.add_attr(self.robot_trans)
            self.robot.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.robot)   
                     
            self.robot_orientation = pyglet_rendering.Line(start=(0.0, 0.0), end=(15.0, 0.0))
            self.robot_orientation.linewidth.stroke = 2
            self.robot_orientation.add_attr(self.robot_trans)
            self.robot_orientation.set_color(0, 0, 0)
            self.viewer.add_geom(self.robot_orientation)
        
        if self.state is None:
            return None
        
        # Maybe we can update the position of the robot and goal
        self.robot_trans.set_translation(to_pixel(self.goal_pos[0], shift=screen_width/2), to_pixel(self.goal_pos[1], shift=screen_height/2))
        self.robot_trans.set_translation(to_pixel(self.state[0], shift=screen_width/2), to_pixel(self.state[1], shift=screen_height/2))
        self.robot_trans.set_rotation(self.state[2])
        
        return self.viewer.render(return_rgb_array=mode == "rgb_array")
        
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _step(self, action):
        # Get the current state
        # This f is always zero is because the unicycle is a driftless system and we are doing +=
        self.state += self.dt * (self.get_f(self.state) + self.get_g(self.state) @ action)
        self.state -= self.dt * -0.1 * self.get_g(self.state) @ np.array([np.cos(self.state[2]),  0]) #* np.random.multivariate_normal(self.disturb_mean, self.disturb_covar, 1).squeeze()
        self.episode_step += 1
        
        # useless here, just to satisfy the interface
        info = dict()
        
        # Update goal position since the goal is moving
        self.goal_pos = [self.circle_r * np.cos(self.speed * np.pi/5 * self.episode_step * self.dt), self.circle_r * np.sin(self.speed * np.pi/5 * self.episode_step * self.dt)]
        
        # Calculate the reward
        dist_goal = self._goal_dist()
        reward = self.last_goal_dist - dist_goal
        self.last_goal_dist = dist_goal
        
        # Check if goal is met
        # Here we only terminate when the episode step when complete 10 seconds
        done = self.episode_step >= 2500
        
        return self.state, reward, done, info
        
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
    
    # Get the distance between the current position and the goal position
    def _goal_dist(self):
        return np.linalg.norm(self.goal_pos - self.state[:2] )