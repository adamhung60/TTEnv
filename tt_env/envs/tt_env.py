import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import pkg_resources
import cv2
import time
import random

class TTEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None):
        self.robot_path = pkg_resources.resource_filename(__name__, 'robot/robot.urdf')
        self.table_path = pkg_resources.resource_filename(__name__, 'table/robot.urdf')
        self.ball_path = pkg_resources.resource_filename(__name__, 'ball/robot.urdf')
        self.max_steps = 250
        self.steps_taken = 0
        self.joints = [0, 1, 2, 3]
        self.numJoints = len(self.joints)
        self.success = False
        self.render_mode = render_mode
        if self.render_mode == "human":
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.8)
        p.setRealTimeSimulation(0)
        p.resetSimulation()

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(35,), dtype=float)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=float)

    def get_success(self):
        return self.success

    def get_obs(self):

        link_positions = []
        link_orientations = []
        joint_positions = []
        joint_velocities = []
        for link in range(self.numJoints-1):
            link_state = p.getLinkState(self.robot, link)
            link_positions += link_state[0]
            link_orientations += link_state[1]
        for joint in self.joints:
            joint_state = p.getJointState(self.robot, joint)
            joint_positions.append(joint_state[0])
            joint_velocities.append(joint_state[1])
        
        ball_position = list(p.getBasePositionAndOrientation(self.ball)[0])
        ball_velocity = list(p.getBaseVelocity(self.ball)[0])

        return np.array(link_positions+link_orientations+joint_positions+joint_velocities+ball_position+ball_velocity)
    
    def is_dead(self):
        return False
    
    def reset(self,seed=None, options = None): 
        super().reset(seed=seed)
        self.steps_taken = 0
        p.resetSimulation()
        p.setGravity(0,0,-9.8)
        self.plane = p.loadURDF("plane.urdf") 
        self.robot = p.loadURDF(self.robot_path, [0,0,0],useFixedBase = 1)
        self.table = p.loadURDF(self.table_path, [0.374,0,0],useFixedBase = 1)
        self.ball = p.loadURDF(self.ball_path, [0.7,0,0.1],useFixedBase = 0)
        p.changeDynamics(self.ball, -1, restitution = 0.9)
        p.changeDynamics(self.table, -1, restitution = 0.9)
        p.changeDynamics(self.robot, 2, restitution = 0.9)
        #p.resetBaseVelocity(self.ball, linearVelocity = [random.uniform(-2.1, -1.3),random.uniform(-0.5, 0.5),1.2])
        p.resetBaseVelocity(self.ball, linearVelocity = [-1.5,0.3,1.2])

        self.state = 0
        self.terminated = False
        self.truncated = False

        observation = self.get_obs()
        info = {"info":"hi"}

        return observation, info
    
    def getReward(self):
        if p.getContactPoints(self.ball, self.plane):
                self.terminated = True
                return 0
        elif self.state == 0: # ball bounces on table
            if p.getContactPoints(self.table, self.ball):
                self.state = 1
                return 0
            elif p.getContactPoints(self.robot, self.ball):
                self.terminated = True
                return -0.5 # robot hits ball before bounce
            else:
                return 0
        elif self.state == 1: # robot hits ball
            if p.getContactPoints(self.robot, self.ball, 3):
                self.state = 2
                return 1 # robot hits ball correctly
            elif p.getContactPoints(self.ball):
                self.terminated = True
                return -0.5 # ball hits something other than paddle
            else:
                return 0
        elif self.state == 2: # ball lands somewhere
            point =  p.getContactPoints(self.table, self.ball)
            if point and point[0][5][0] > 0.374:
                self.terminated = True
                self.success = True
                return 10 # ball goes in
            elif p.getContactPoints(self.ball):
                self.terminated = True
                return -0.5 # something else happens
            else:
                return 0
        
    def step(self, action):
        
        action = action*5
        p.setJointMotorControlArray(self.robot,self.joints, controlMode=p.VELOCITY_CONTROL, forces = [0]*self.numJoints)
        p.setJointMotorControlArray(
            bodyUniqueId = self.robot, 
            jointIndices = self.joints, 
            controlMode = p.TORQUE_CONTROL, 
            forces = list(action))
        
        p.stepSimulation()
        self.steps_taken += 1
        if self.steps_taken >= self.max_steps:
            self.truncated = True

        dead = self.is_dead()
        if dead:
            self.terminated = True

        observation = self.get_obs()

        reward = self.getReward()

        if self.render_mode == "human":
            self.render_frame()

        info = {"info":"hi"}

        return observation, reward, self.terminated, self.truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self.render_frame()

    def render_frame(self):
        focus_position,_ = p.getBasePositionAndOrientation(self.robot)
        focus_position = tuple([focus_position[0] + 0.2,focus_position[1] + 0.2,focus_position[2]])
        p.resetDebugVisualizerCamera(
            cameraDistance=0.8, 
            cameraYaw = 60, 
            cameraPitch = -20, 
            cameraTargetPosition = focus_position
        )
        if self.render_mode == "human":
            time.sleep(0.0005)
        if self.render_mode == "rgb_array":
            h,w = 4000, 4000
            image = np.array(p.getCameraImage(h, w)[2]).reshape(h,w,4)
            image = image[:, :, :3]
            image = cv2.convertScaleAbs(image)
            return image
        
    def close(self):
        return