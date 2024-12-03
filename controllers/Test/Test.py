# Add the controller Webots Python library path
import sys
webots_path = 'C:\Program Files\Webots\lib\controller\python'
sys.path.append(webots_path)

# Add Webots controlling libraries
from controller import Robot
from controller import Supervisor

import random
# Some general libraries
import os
import time
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from collections import namedtuple, deque
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 0.01              # for soft update of target parameters
LR =0.01       # learning rate 
UPDATE_EVERY = 4       # how often to update the network

# Create an instance of robot
robot = Robot()

# Seed Everything
seed = 42
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Environment(Supervisor):
    """The robot's environment in Webots."""
    
    def __init__(self):
        super().__init__()
                
        # General environment parameters
        self.max_speed = 1.5 # Maximum Angular speed in rad/s
        self.destination_coordinate = np.array([-0.03,2.73]) # Target (Goal) position
        self.reach_threshold = 0.06 # Distance threshold for considering the destination reached.
        obstacle_threshold = 0.1 # Threshold for considering proximity to obstacles.
        self.obstacle_threshold = 1 - obstacle_threshold
        self.floor_size = np.linalg.norm([8, 8])
        
        
        # Activate Devices
        #~~ 1) Wheel Sensors
        self.left_motor = robot.getDevice('left wheel')
        self.right_motor = robot.getDevice('right wheel')

        # Set the motors to rotate for ever
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        # Zero out starting velocity
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        
        #~~ 2) GPS Sensor
        sampling_period = 1 # in ms
        self.gps = robot.getDevice("gps")
        self.gps.enable(sampling_period)
        
        #~~ 3) Enable Touch Sensor
        self.touch = robot.getDevice("touch sensor")
        self.touch.enable(sampling_period)
        #~~ 4) compas Sensor
        sampling_period = 1 # in ms
        self.compass = robot.getDevice("compass")
        self.compass.enable(sampling_period)
        # List of all available sensors
        available_devices = list(robot.devices.keys())
        # Filter sensors name that contain 'so'
        filtered_list = [item for item in available_devices if 'so' in item and any(char.isdigit() for char in item)]
        filtered_list = sorted(filtered_list, key=lambda x: int(''.join(filter(str.isdigit, x))))

        # Reset
        #self.simulationReset()
        #self.simulationResetPhysics()
        #super(Supervisor, self).step(int(self.getBasicTimeStep()))
        #robot.step(200) # take some dummy steps in environment for initialization
        
        # Create dictionary from all available distance sensors and keep min and max of from total values
        self.max_sensor = 0
        self.min_sensor = 0
        self.dist_sensors = {}
        for i in filtered_list:    
            self.dist_sensors[i] = robot.getDevice(i)
            self.dist_sensors[i].enable(sampling_period)
            self.max_sensor = max(self.dist_sensors[i].max_value, self.max_sensor)    
            self.min_sensor = min(self.dist_sensors[i].min_value, self.min_sensor)
           
    def getFacing(self):
        robot.step(500) 
        gps_value = self.gps.getValues()[0:3]
        current_coordinate = np.array(gps_value)
        dv = [self.destination_coordinate[1]-current_coordinate[1],
        self.destination_coordinate[0]-current_coordinate[0],
        0.001-current_coordinate[2]
        ]
        dv = np.array(dv)
        robot.step(500) 
        x=self.compass.getValues()
        return np.dot(dv,x)  
        
    def normalizer(self, value, min_value, max_value):
        """
        Performs min-max normalization on the given value.

        Returns:
        - float: Normalized value.
        """
        normalized_value = (value - min_value) / (max_value - min_value)        
        return normalized_value
        

    def get_distance_to_goal(self):
        """
        Calculates and returns the normalized distance from the robot's current position to the goal.
        
        Returns:
        - numpy.ndarray: Normalized distance vector.
        """
        
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        distance_to_goal = np.linalg.norm(self.destination_coordinate - current_coordinate)
        normalizied_coordinate_vector = self.normalizer(distance_to_goal, min_value=0, max_value=self.floor_size)
        
        return normalizied_coordinate_vector
        
    
    def get_sensor_data(self):
        """
        Retrieves and normalizes data from distance sensors.
        
        Returns:
        - numpy.ndarray: Normalized distance sensor data.
        """
        
        # Gather values of distance sensors.
        sensor_data = []
        for z in self.dist_sensors:
            sensor_data.append(self.dist_sensors[z].value)  
            
        sensor_data = np.array(sensor_data)
        normalized_sensor_data = self.normalizer(sensor_data, self.min_sensor, self.max_sensor)
        
        return normalized_sensor_data
        
    
    def get_observations(self):
        """
        Obtains and returns the normalized sensor data and current distance to the goal.
        
        Returns:
        - numpy.ndarray: State vector representing distance to goal and distance sensors value.
        """
        
        normalized_sensor_data = np.array(self.get_sensor_data(), dtype=np.float32)
        normalizied_current_coordinate = np.array([self.get_distance_to_goal()], dtype=np.float32)
        gps_value = self.gps.getValues()[0:2]
        current_coordinate = np.array(gps_value)
        Orient = np.array([self.getFacing()])
        #gps_value = self.gps.getValues()[0:2]
        #current_coordinate = np.array(gps_value)
        #x_distance =np.array([self.destination_coordinate[0]-current_coordinate[0]], dtype=np.float32)
        #y_distance=np.array([self.destination_coordinate[1]-current_coordinate[1]], dtype=np.float32)
        state_vector = np.concatenate([normalizied_current_coordinate, normalized_sensor_data,Orient], dtype=np.float32)
        return state_vector
    
    
    def reset(self):
        """
        Resets the environment to its initial state and returns the initial observations.
        
        Returns:
        - numpy.ndarray: Initial state vector.
        """
        self.simulationReset()
        self.simulationResetPhysics()
        super(Supervisor, self).step(int(self.getBasicTimeStep()))
        return self.get_observations()


    def step(self, action, max_steps):    
        """
        Takes a step in the environment based on the given action.
        
        Returns:
        - state       = float numpy.ndarray with shape of (3,)
        - step_reward = float
        - done        = bool
        """
        
        self.apply_action(action)
        step_reward, done = self.get_reward()
        
        state = self.get_observations() # New state
        
        # Time-based termination condition
        if (int(self.getTime()) + 1) % max_steps == 0:
            done = True
                
        return state, step_reward, done
        

    def get_reward(self):
        """
        Calculates and returns the reward based on the current state.
        
        Returns:
        - The reward and done flag.
        """
        
        done = False
        reward = 0
        
        normalized_sensor_data = self.get_sensor_data()
        normalized_current_distance = self.get_distance_to_goal()
        
        normalized_current_distance *= 100 # The value is between 0 and 1. Multiply by 100 will make the function work better
        reach_threshold = self.reach_threshold * 100
        # (1) Reward according to distance 
        if normalized_current_distance < 50:
            if normalized_current_distance < 10:
                growth_factor = 10
                A = 8
            if normalized_current_distance < 20:
                growth_factor = 8
                A = 5
            elif normalized_current_distance < 30:
                growth_factor = 6
                A = 3.5
            elif normalized_current_distance < 40:
                growth_factor = 4
                A = 2.5
            else:
                growth_factor = 1.2
                A = 0.9
            reward += A * (1 - np.exp(-growth_factor * (1 / normalized_current_distance)))
            
        else: 
            reward += -normalized_current_distance / 100
            

        # (2) Reward or punishment based on failure or completion of task
        check_collision = self.touch.value
        if normalized_current_distance < reach_threshold:
            # Reward for finishing the task
            done = True
            reward += 25
            print('+++ SOlVED +++')
        elif check_collision:
            # Punish if Collision
            print("Collision happened")
            done = True
            reward -= normalized_current_distance/10
            
            
        # (3) Punish if close to obstacles
        elif np.any(normalized_sensor_data[normalized_sensor_data > self.obstacle_threshold]):
            reward -= 1
       
        # (4)check orient
        x = self.getFacing()
        if x>normalized_current_distance*10:
            reward +=0.1
        elif x<-1*(normalized_current_distance*10):
            reward -=0.1
        elif x<=normalized_current_distance*10 and x>normalized_current_distance*7:
            reward +=0.05
        elif x>=-1*normalized_current_distance*10 and x>-1*(normalized_current_distance*7):
            reward -=0.05
        # (5) Reward or punishment based on axis distance
        #gps_value = self.gps.getValues()[0:2]
        #current_coordinate = np.array(gps_value)
        #x_distance =np.array(self.destination_coordinate[0]-current_coordinate[0])
        #y_distance=np.array(self.destination_coordinate[1]-current_coordinate[1])
        #reach_threshold=reach_threshold/10
        #if x_distance < reach_threshold and y_distance < reach_threshold:
          #   reward += 1
        #elif x_distance > reach_threshold and y_distance < reach_threshold:
         #    reward -= 0.001
        #elif x_distance < reach_threshold and y_distance > reach_threshold:
         #    reward -= 0.01
        #elif x_distance > reach_threshold and y_distance > reach_threshold:
         #    reward -= 0.1
        
        return reward, done


    def apply_action(self, action):
        """
        Applies the specified action to the robot's motors.
        
        Returns:
        - None
        """
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        
        if action == 0: # move forward
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        elif action == 1: # turn right
            self.left_motor.setVelocity(self.max_speed)
            self.right_motor.setVelocity(-self.max_speed)
        elif action == 2: # turn left
            self.left_motor.setVelocity(-self.max_speed)
            self.right_motor.setVelocity(self.max_speed)
        
        robot.step(500)
        
        self.left_motor.setPosition(0)
        self.right_motor.setPosition(0)
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)           

    
class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=10, fc2_units=8,fc3_units=6):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            fc3_units (int): Number of nodes in third hidden layer
            fc4_units (int): Number of nodes in forth hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)
        
class Agent_REINFORCE():
    """Agent implementing the REINFORCE algorithm."""

    def __init__(self, save_path, load_path, num_episodes, max_steps, 
                    state_size,action_size,seed):
                
        self.save_path = save_path
        self.load_path = load_path
        
        os.makedirs(self.save_path, exist_ok=True)
        
        # Hyper-parameters Attributes
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
        # Create the self.optimizers
        
        # instance of env
        self.env = Environment()
               
        
    def save(self, path):
        """Save the trained model parameters after final episode and after receiving the best reward."""
        torch.save(self.qnetwork_local.state_dict(), self.save_path + path)
    
    
    def load(self):
        """Load pre-trained model parameters."""
        self.qnetwork_local.load_state_dict(torch.load(self.load_path))


    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        else:
            a =action_values.cpu().data.numpy()[0]
            return np.argmax(a)
        
    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        self.qnetwork_local.train()
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


    def dqn(self, eps_start=1.0, eps_end=0.1, eps_decay=0.995):
        """
        Train the agent using the REINFORCE algorithm.
        
        This method performs the training of the agent using the REINFORCE algorithm. It iterates
        over episodes, collects experiences, computes returns, and updates the policy network.
        """           
        start_time = time.time()
        reward_history = []
        best_score = -np.inf
        eps = eps_start 

        for episode in range(1, self.num_episodes+1):
            done = False
            state = self.env.reset()
            rewards = []
            scores = []   
            scores_window = deque(maxlen=100)
            ep_reward = 0
            while True:
                action = agent.act(state, eps)
                next_state, reward, done = self.env.step(action.item(), self.max_steps)
                self.step(state, action, reward, next_state, done) 
                state = next_state
                rewards.append(reward)
                ep_reward += reward
                if done:
                    reward_history.append(ep_reward)         
                    if ep_reward > best_score:
                        self.save(path='/best_weights.pt')
                        best_score = ep_reward
                    print(f"Episode {episode}: Score = {ep_reward:.3f}")
                    break
            eps = max(eps_end, eps_decay*eps) # decrease epsilon
                    
        # Save final weights and plot reward history
        self.save(path='/final_weights3.pt')
        self.plot_rewards(reward_history)        
                
        # Print total training time
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
        
              
    def test(self):
        """
        Test the trained agent.
        This method evaluates the performance of the trained agent.
        """
        
        start_time = time.time()
        rewards = []
        self.load()
        self.qnetwork_local.eval()
        for episode in range(1, 15):
            state = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                with torch.no_grad():
                    action_values = self.qnetwork_local(torch.from_numpy(state).float().unsqueeze(0).to(device))
                action = np.argmax(action_values.cpu().data.numpy()[0])
                state, reward, done = self.env.step(action.item(), self.max_steps) 
                ep_reward += reward
            rewards.append(ep_reward)
            print(f"Episode {episode}: Score = {ep_reward:.3f}")
        print(f"Mean Score = {np.mean(rewards):.3f}")
        
        elapsed_time = time.time() - start_time
        elapsed_timedelta = timedelta(seconds=elapsed_time)
        formatted_time = str(elapsed_timedelta).split('.')[0]
        print(f'Total Spent Time: {formatted_time}')
    
    def testNetworks(self):
        firstModel = QNetwork(state_size, action_size, seed).to(device)
        firstModel.load_state_dict(torch.load('./results/first_weights.pt'))
        print("firstModel")
        for local_param in  firstModel.parameters():
            print(local_param.data)
        secondModel= QNetwork(state_size, action_size, seed).to(device)
        secondModel.load_state_dict(torch.load('./results/final_weights.pt'))
        print("secondModel")
        for local_param in  secondModel.parameters():
            print(local_param.data)
    def plot_rewards(self, rewards):
        # Calculate the Simple Moving Average (SMA) with a window size of 25
        sma = np.convolve(rewards, np.ones(25)/25, mode='valid')
        
        plt.figure()
        plt.title("Episode Rewards")
        plt.plot(rewards, label='Raw Reward', color='#142475', alpha=0.45)
        plt.plot(sma, label='SMA 25', color='#f0c52b')
        plt.xlabel("Episode")
        plt.ylabel("Rewards")
        plt.legend()
        
        plt.savefig(self.save_path + '/reward_plot3.png', format='png', dpi=1000, bbox_inches='tight')
        plt.tight_layout()
        plt.grid(True)
        plt.show()
        plt.clf()
        plt.close()
            
def stopping_criterion(env):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return(int(env.getTime()) + 1) % max_steps == 0

if __name__ == '__main__':
    # Parameters
    save_path = './results'   
    load_path = './results/final_weights.pt'

    num_episodes = 2500 
    max_steps = 200
    state_size=4
    action_size=3
    seed=0
    agent = Agent_REINFORCE(save_path, load_path, num_episodes, max_steps, 
                             state_size,action_size,seed)
    #agent.testNetworks()
    #agent.dqn()
    #agent.testNetworks()
    agent.test()    