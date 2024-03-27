# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
#Variable allows the access of dynamic graphs, which ensures very fast computation of gradient
from torch.autograd import Variable

# Creating the architecture of the Neural Network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        #initialising the input_size and nb_action(output size)
        self.input_size = input_size 
        self.nb_action = nb_action
        #creating the first fully connected layers (inputlayer to hidden layer), with 30 hidden neurons
        self.fc1 = nn.Linear(input_size, 30)
        #creating the second fully connected layers (seond hidden layer)
        self.fc2 = nn.Linear(30, 60)
        #creating the third fully connected layers (hidden layer- output layer)
        self.fc3 = nn.Linear(60, nb_action)
    
    def forward(self, state):
        #fc1 is activated by relu 
        x = F.relu(self.fc1(state))
        #fc2 is activated by relu 
        x = F.relu(self.fc2(x))
        #q-values are reutrned by fc2
        q_values = self.fc3(x)
        return q_values

# Implementing Experience Replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        #initiate the capcaity varaible which basically measures the capcity of the list tranasitions in our memory
        self.capacity = capacity
        #empty memory
        self.memory = []
    
    def push(self, event):
        self.memory.append(event)
        #When we append a new item to the memory list and the length of memory seems to esxceed the capacity, we remove the first item
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #in this code we sample 100 transitions from the replay memory
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        #discout factor of the Q-learning algorithm
        self.gamma = gamma 
        #list consisting of the reward gained/ reduced
        self.reward_window = [] 
        #input_size is 5 (3 signals and the two orientations, nb_action is 3(straight, left,right))
        self.model = Network(input_size, nb_action) 
        #Instantiate Replay memory object with capacity of 100_000 (meaning it will learn 100_000 different transitions)
        self.memory = ReplayMemory(100000) 
        #Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001) 
        #last_state variable is the last transition from the list of transitions in our replay memory, and this value is initiated to a tensor of dim (1, input_size), the first dimension to represent the batch, this is initialised to this:
        self.last_state = torch.Tensor(input_size).unsqueeze(0) 
        #this is the last action in the transition, first initialised to 0
        self.last_action = 0 
        # this is the last reward in list of transitions, first initiliased to 0
        self.last_reward = 0
    
    def select_action(self, state):
        # We are passing the softmax function to the values predicted by the model (forward propagation), we are multiplying by 100, value known as temperature, to give the agent some certainty to its action
        probs = F.softmax(self.model(Variable(state, volatile = True))*300) 
        # picks a random action as per the softmax probabilites
        action = probs.multinomial(num_samples=1)
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        #this whole 'gather' method basically selects the q-values corresponding to the action in the batch_action
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1) 
        #we select the maximum q-value for each of the states, i.e max(Q(s, a)) where s is at timestamp t+1
        next_outputs = self.model(batch_next_state).detach().max(1)[0] 
        #calculating the target Q-value by plugging in the respective values we calculated earlier
        target = self.gamma*next_outputs + batch_reward 
        #calculates the temporal difference loss
        td_loss = F.smooth_l1_loss(outputs, target)
        #sets all the gradients of weights to zero
        self.optimizer.zero_grad()
        #backpropagates the loss
        td_loss.backward(retain_graph = True)
        #performing the weight updates
        self.optimizer.step()
    
    def update(self, reward, new_signal):
        #we get the signal detected by the car, and we pass this as the new_state to our model
        new_state = torch.Tensor(new_signal).float().unsqueeze(0) 
        #pushing all the required inputs to the memory
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        #select future action from the new_state
        action = self.select_action(new_state)
        #if the memory length is greater than 100, we sample it from the memory object, and we pass it to the learn function to update the weights
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            #we pass it to the learn method, for the calculation of loss for backpropagation
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #we update the last_action, last_state, last_reward 
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        #we append to the reward window
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        #finally return the action
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')
    
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
