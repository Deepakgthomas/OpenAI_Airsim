## Taken from - https://github.com/ashdtu/openai_drone_gym/blob/master/drone_gym/envs/drone_airsim.py

import gym
from gym import error, spaces
import airsim
import numpy as np
import math
import time
from PIL import Image

class DroneAirsim(gym.Env):
        # Constructor
	def __init__(self, image_shape=[256,256], hover_height = -1.2, pose_offset=[0,0]):

		self.z = hover_height # set the height to hover_height parameter (default -1.2)
		self.client = airsim.MultirotorClient() # Initialize client
		self.client.confirmConnection() # Waits for connection
		self.client.enableApiControl(True) # Enables API Control
		self.client.armDisarm(True) # Disarm arm
		self.action_duration = 1 # Actions last for 1
		self.num_frames = 4 # Number of frames being looked at
		self.img_shape = image_shape # Sets the shape of the image (default 256x256 pixels)
		self.action_space = spaces.Discrete(4) # 4 possible actions
		self.action_dim = self.action_space.n # Sets dimension of actions
		self.pose_offset = np.array(pose_offset) # Sets pose offset (default [0, 0])
		self.start_pose = np.array([0.0, 0.0]) + self.pose_offset # Sets start_pose
		self.observation_space = spaces.Box(low=0, high=255,shape=[self.num_frames,self.img_shape[0],self.img_shape[1]],dtype=np.uint8) # Sets observation space

        # Function to go straight
	def straight(self,speed, reverse):
		pitch, roll, yaw = airsim.to_eularian_angles(self.client.simGetVehiclePose().orientation) # Gets pitch, roll, and yaw in current state
		dx = math.cos(yaw) * speed * reverse # Sets dx
		dy = math.sin(yaw) * speed * reverse # Sets dy
		x = self.client.simGetVehiclePose().position.x_val # Gets x position at current state
		y = self.client.simGetVehiclePose().position.y_val # Gets y position at current state
		self.client.moveToPositionAsync(x + dx, y + dy, self.z,speed,airsim.DrivetrainType.ForwardOnly) # Moves the drone to the new position based off of dx, dy, x, and y
		init_time=time.time() # Return init_time
		return init_time

        # Function to yaw right
	def yaw_right(self):
		self.client.rotateByYawRateAsync(self.yaw_degrees, self.action_duration) # Rotates the drone
		init_time = time.time() # Return init_time
		return init_time

        # Function to yaw left
	def yaw_left(self):
		self.client.rotateByYawRateAsync(-self.yaw_degrees, self.action_duration) # Rotates the drone
		init_time = time.time() # Return init_time
		return init_time

        # Takes a certain action based on action parameter
        # action=0 => forward
        # action=1 => yaw right
        # action=2 => yaw left
        # action=3 => reverse
	def take_action(self, action):

		collided = False # Assume not collided
	        frame_buffer=[]	 # Collects all frames captured while doing action
		prev_pose=self.getPose() # Gets the previous pose

		if action == 0:
			start=self.straight(1,1) # Move in direction of yaw heading with 1m/s for 1s

                        # "Simulates" the action
			while self.action_duration > time.time() - start:
                                
                                # Check if a collision happened during the action
				if self.client.simGetCollisionInfo().has_collided:
					collided=True

                                # Append the current frame to the frame buffer
                                frame_buffer.append(self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)])[0])

			self.client.moveByVelocityZAsync(0, 0, self.z,1).join()

		if action == 1:
			# Rotate right on z axis for 1 sec
			start = self.yaw_right()

                        # "Simulates" the action
			while self.action_duration > time.time() - start:
                                
                                # Check if a collision happened during the action
				if self.client.simGetCollisionInfo().has_collided:
					collided=True

                                # Append the current frame to the frame buffer
				frame_buffer.append(self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)])[0])

			self.client.moveByVelocityZAsync(0, 0, self.z, 0.5).join()
			self.client.rotateByYawRateAsync(0,0.5).join()

		if action == 2:
			# Rotate left on z axis  for 1s
			start= self.yaw_left()


			while self.action_duration > time.time() - start:

                                # Check if a collision happened during the action
				if self.client.simGetCollisionInfo().has_collided:
					collided=True

                                # Append the current frame to the frame buffer
				frame_buffer.append(self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)])[0])

			self.client.moveByVelocityZAsync(0, 0, self.z,0.5).join()
			self.client.rotateByYawRateAsync(0, 0.5).join()

		if action == 3:
                        # Goes in reverse for 1 sec
			start = self.straight(1, -1)
                        
                        # "Simulates" the action
			while self.action_duration > time.time() - start:

                                # Check if a collision happened during the action
				if self.client.simGetCollisionInfo().has_collided:
					collided = True
                                        
                                # Append the current frame to the frame buffer
				frame_buffer.append(self.client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0])

			self.client.moveByVelocityZAsync(0, 0, self.z, 1).join()

                # Return the original pose before the action, the frame buffer, and if the drone collided or not
		return prev_pose,frame_buffer,collided

        # Function that preprocesses the frame by reshaping it to appropriate dimensions
	def process_frame(self,response):
                
		frame = airsim.string_to_uint8_array(response.image_data_uint8) # Converts frame to unsigned 8 bit int array
		try:
			frame = frame.reshape(self.img_shape[0],self.img_shape[1], 3) # Reshape the frame to what the shape should be (set in the constructor)
		except ValueError:						      # Bug with client.simGetImages:randomly drops Image response sometimes
			frame = np.zeros((self.img_shape[0],self.img_shape[1]))
			#self.dropped_frames += 1
			return frame

                # Converts back to a processable frame and returns it
		frame = Image.fromarray(frame).convert('L')
		frame = np.asarray(frame) 
		return frame

	def stackFrames(self, *args, init_state=False):
                
		if init_state:
                        # Get latest frame
			response = self.client.simGetImages([
				airsim.ImageRequest("0", airsim.ImageType.Scene,False,False)
			])[0]
                        
                        # Print shape of the frame
			print("Images = ", self.img_shape[0], "Images = ", self.img_shape[1])

			#assert response.height == self.img_shape[0] and response.width == self.img_shape[1], "Input Image size from airsim settings.json and env doesn't match"


                        # Reshape the frame appropriately
			frame=self.process_frame(response)

                        # Sets all stack frames to the latest frame
			stack_frames=[frame for i in range(self.num_frames)]
                        
                        # Converts stack_frames to array
			stack_frames=np.array(stack_frames)
			return stack_frames

		else:
                        # Creates a custom stack_frame based on *args input
			responses_in=args[0]
			len_frames=len(responses_in)

			try:
				assert len_frames >= self.num_frames+2, "Frame rate not enough"
			except AssertionError:
				stack_frames = np.zeros((self.num_frames,self.img_shape[0],self.img_shape[1]))
				#self.dropped_frames += 1
				self.reset()
				return stack_frames

			"""
			0, N/2-2, N/2, N/2+2, N-1 frames are stacked together to get state
			Modify these indexes if frame rate is less(say 5-7 fps)
			"""
			indexes=[0,int(len_frames/2)-2,int(len_frames/2),int(len_frames/2)+2,len_frames-1]
			stack_frames=[self.process_frame(responses_in[i]) for i in indexes]
			stack_frames=np.array(stack_frames)
			return stack_frames

	def get_reward(self,collision,prev_pose,new_pose):

		eps_end=False # Determines whether to end or not

		if not collision:
			prev_dist=np.sqrt(np.power((self.goal[0]-prev_pose[0]),2) + np.power((self.goal[1]-prev_pose[1]),2)) # Get the previous distance
			new_dist=np.sqrt(np.power((self.goal[0]-new_pose[0]),2) + np.power((self.goal[1]-new_pose[1]),2)) # Get the new distance

			if new_dist < 3:
				eps_end=True # End
				reward=100 # Reward 100 for lasting 3 units
			else:
				eps_end=False # Continue the simulation
				reward=-1+(prev_dist-new_dist) # Reward based on the difference between prev_dist and new_dist
                # If there was a collision
		else:
			eps_end=True # End
			reward=-100 # Reward -100 for crashing

		return reward,eps_end

	def getPose(self):
                # Gets the true pose with current position+pose_offset (set in ctor)
		return np.array([self.client.simGetVehiclePose().position.x_val,self.client.simGetVehiclePose().position.y_val]) + self.pose_offset


        # Resets simulation
	def reset(self):
		self.client.reset()
		self.client.enableApiControl(True)
		self.client.armDisarm(True)
		self.client.moveToZAsync(self.z,1).join()
		return self.stackFrames(init_state=True)

	def render(self, mode='human'):
		raise NotImplementedError

        # Shuts down environment
	def close(self):
		print("Shutting down environment...")
		self.client.armDisarm(False)
		self.client.reset()
		self.client.enableApiControl(False)

	























