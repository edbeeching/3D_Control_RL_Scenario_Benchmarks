#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 10:53:06 2018

@author: anonymous
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from masked_gru import MaskedGRU
from distributions import Categorical    


# A temporary solution from the master branch.
# https://github.com/pytorch/pytorch/blob/7752fe5d4e50052b3b0bbc9109e599f8157febc0/torch/nn/init.py#L312
# Remove after the next version of PyTorch gets release. 
# TODO: test the official PyTorch orthogonal init implementation 0.4.1
def orthogonal(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = torch.Tensor(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    q, r = torch.qr(flattened)
    # Make Q uniform according to https://arxiv.org/pdf/math-ph/0609050.pdf
    d = torch.diag(r, 0)
    ph = d.sign()
    q *= ph.expand_as(q)

    if rows < cols:
        q.t_()

    tensor.view_as(q).copy_(q)
    tensor.mul_(gain)
    return tensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        print(m.__class__.__name__)
        print(m.weight.size())
        if m.bias is not None:
            m.bias.data.fill_(0)

class Lin_View(nn.Module):
	def __init__(self):
		super(Lin_View, self).__init__()
        
	def forward(self, x):
		return x.view(x.size()[0], -1)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, 
            deterministic=False):

        value, x, states = self(inputs, states, masks)
 
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
 
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, 
                         masks, actions):
        
        value, x, states  = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states
    
    def get_action_value_and_probs(self, inputs, states, masks, 
                                   deterministic=False):
        
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, F.softmax(self.dist(x),dim=1), states


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, input_shape, params):
        super(CNNPolicy, self).__init__()

  
        self.conv_head = nn.Sequential(nn.Conv2d(num_inputs, params.conv1_size, 8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(params.conv1_size, params.conv2_size, 4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(params.conv2_size, params.conv3_size, 3, stride=1),
                                        nn.ReLU(True))

        conv_input = torch.Tensor(torch.randn((1,) + input_shape))
        print(conv_input.size(), self.conv_head(conv_input).size(), self.conv_head(conv_input).size())
        self.conv_out_size = self.conv_head(conv_input).nelement()    
        self.hidden_size = params.hidden_size
        

        self.linear1 = nn.Linear(self.conv_out_size, self.hidden_size)

        if params.recurrent_policy:
            #self.gru = MaskedGRU(self.hidden_size, self.hidden_size) TODO: check speedup with masked GRU optimization
            self.gru = nn.GRUCell(self.hidden_size, self.hidden_size)

        self.critic_linear = nn.Linear(self.hidden_size, 1)
        self.dist = Categorical(self.hidden_size, params.num_actions)

        self.params = params
        self.train()
        self.reset_parameters()
        
    @property
    def state_size(self):
        if hasattr(self, 'gru'):
            return self.hidden_size
        else:
            return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        for i in range(0, 6, 2):
            self.conv_head[i].weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)
        

        if hasattr(self, 'gru'):
            #self.gru.reset_parameters()
            orthogonal(self.gru.weight_ih.data)
            orthogonal(self.gru.weight_hh.data)
            self.gru.bias_ih.data.fill_(0)
            self.gru.bias_hh.data.fill_(0)
                 
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)
             
        
    def forward(self, inputs, states, masks):
        x = self.conv_head(inputs * (1.0/255.0))
        x = x.view(-1, self.conv_out_size)
        x = self.linear1(x)
        x = F.relu(x)
        if hasattr(self, 'gru'):
            #x, states = self.gru(x, states, masks)
            if inputs.size(0) == states.size(0):
                x = states = self.gru(x, states * masks)
            else:
                x = x.view(-1, states.size(0), x.size(1))
                masks = masks.view(-1, states.size(0), 1)
                outputs = []
                for i in range(x.size(0)):
                    hx = states = self.gru(x[i], states * masks[i])
                    outputs.append(hx)
                x = torch.cat(outputs, 0)
     
        return self.critic_linear(x), x, states

if __name__ == '__main__':
    
    from arguments import parse_game_args 
    params = parse_game_args()  
    params.num_actions = 5
    
    model = CNNPolicy(3, (3, 64, 112), params)
    example_input = torch.randn(1,3,64,112)
    out = model.conv_head[:4](example_input)
    print(out.size())
