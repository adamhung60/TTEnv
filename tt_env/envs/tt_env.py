import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
import cv2
import time
import random
import math

class TTEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, y = 0):
        self.render_mode = render_mode
        self.robot_path = pkg_resources.resource_filename(__name__, 'robot/robot.urdf')
        self.table_path = pkg_resources.resource_filename(__name__, 'table/robot.urdf')
        self.ball_path = pkg_resources.resource_filename(__name__, 'ball/robot.urdf')

        self.max_steps = 500
        self.steps_taken = 0

        self.joints = [0, 1, 2, 3, 4]
        self.numJoints = len(self.joints)
        
        self.y = y

        # monitor vars
        self.episode_count = 0
        self.ball_in_count = 0
        self.ball_touch_count = 0
        self.d2t_sum = 0

        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.resetSimulation()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(64,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.numJoints,), dtype=float)

    def get_obs(self):
        joint_positions = [p.getJointState(self.robot, joint)[0] for joint in self.joints]
        ball_position = list(p.getBasePositionAndOrientation(self.ball)[0])
        current_obs = np.array(joint_positions+ball_position)

        self.obs = self.obs[1:]
        self.obs = np.vstack((self.obs, current_obs))

        return self.obs.flatten()
    
    def reset(self,seed=None, options = None): 
        super().reset(seed=seed)
        self.terminated = False
        self.truncated = False
        self.steps_taken = 0
        self.state = 0
        self.obs = np.zeros((8, 8))

        self.episode_count += 1

        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.plane = p.loadURDF("plane.urdf") 
        self.robot = p.loadURDF(self.robot_path, [-1.5,0,0], useFixedBase = 1)
        self.table = p.loadURDF(self.table_path, [0,0,0],useFixedBase = 1)

        start, v = self.get_trajectory()
        self.ball = p.loadURDF(self.ball_path, start)
        p.resetBaseVelocity(self.ball, linearVelocity = v)
        #self.ball = p.loadURDF(self.ball_path, [0.3,0,0.1],useFixedBase = 0)
        #p.resetBaseVelocity(self.ball, linearVelocity = [-1.3,0.2,1.4])

        p.changeDynamics(self.ball, -1, restitution = 0.9)
        p.changeDynamics(self.table, -1, restitution = 0.9)
        p.changeDynamics(self.robot, 2, restitution = 0.9)

        observation = self.get_obs()
        info = {"info":"hi"}

        return observation, info
    
    def get_trajectory(self):
        #goal = [random.uniform(-1.37, 0), random.uniform(-0.7625, 0.7625), 0.1]
        #start = [random.uniform(0, 2.0), random.uniform(-0.7625, 0.7625), random.uniform(0.1, 1)]
        y = random.uniform(-self.y, self.y)
        start = [1.5, 0, 0.5]
        goal = [-1.0, y, 0.1]
        
        #vz = random.uniform(3, 10)
        vz = 3
        t = (vz + math.sqrt(vz**2 + 19.6*(start[2]+goal[2])))/9.8
        vx = (goal[0] - start[0])/t
        vy = (goal[1] - start[1])/t
        v = [vx,vy,vz]
        return start, v

    def get_reward(self):

        if self.state == 0: # ball bounces on table
            if p.getContactPoints(self.table, self.ball):
                self.state = 1
        elif self.state == 1: # robot hits ball
            if p.getContactPoints(self.robot, self.ball, 3):
                self.state = 2
                self.ball_touch_count += 1
                return 1 # robot hits ball correctly
            elif p.getContactPoints(self.ball):
                self.terminated = True
        elif self.state == 2: # ball lands
            point = p.getContactPoints(self.ball, self.plane) or p.getContactPoints(self.ball, self.table)
            if point:
                point = p.getContactPoints(self.ball)
                d2t = self.d2t(point[0][5][0],point[0][5][1])
                self.d2t_sum += d2t
                self.terminated = True
                if d2t == 0:
                    self.ball_in_count += 1
                return 1/(d2t+0.01)
        return 0
    
    def d2t(self,x,y): # distance to table
        if x < 0:
            dx = -x
        elif x > 1.37:
            dx = x - 1.37
        else:
            dx = 0
        
        if y > 0.7625:
            dy = y - 0.7625
        elif y < -0.7625:
            dy = 0.7625 - y
        else:
            dy = 0
            
        return math.sqrt(dx**2 + dy**2)
        
    def step(self, action):
        if self.episode_count >= 1000:
            print("balls in / 1000 = ", self.ball_in_count)
            print("balls touched / 1000 = ", self.ball_touch_count)
            print("average d2t for touched balls = ", self.d2t_sum/(self.ball_touch_count+0.0001))
            self.episode_count = 0
            self.ball_in_count = 0
            self.ball_touch_count = 0
            self.d2t_sum = 0
        
        actions = action*20
        
        p.setJointMotorControlArray(
            bodyUniqueId = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.VELOCITY_CONTROL, 
            targetVelocities = actions,
            forces = [500]*self.numJoints)
        
        p.stepSimulation()
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.truncated = True

        observation = self.get_obs()

        ball_reward = self.get_reward()
        #control_cost = np.sum(np.square(actions))
        reward = ball_reward#-control_cost

        if self.render_mode == "human":
            self.render_frame()

        info = {"info":"hi"}

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        """focus_position,_ = p.getBasePositionAndOrientation(self.robot)
        focus_position = tuple([focus_position[0] + 0.2,focus_position[1] + 0.2,focus_position[2]])
        p.resetDebugVisualizerCamera(
            cameraDistance=0.9, 
            cameraYaw = 40, 
            cameraPitch = -12, 
            cameraTargetPosition = focus_position
        )"""
        if self.render_mode == "human":
            time.sleep(0.000)
        if self.render_mode == "rgb_array":
            h,w = 4000, 4000
            image = np.array(p.getCameraImage(h, w)[2]).reshape(h,w,4)
            image = image[:, :, :3]
            image = cv2.convertScaleAbs(image)
            return image
        
    def close(self):
        return
