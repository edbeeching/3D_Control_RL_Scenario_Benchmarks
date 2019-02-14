#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 11:59:21 2018

@author: anonymous
"""
import os
from collections import deque

import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import torch

from doom_a2c.arguments import parse_game_args
from doom_a2c.models import CNNPolicy, EgoMap0_Policy
from environments import DoomEnvironment
from doom_evaluation_multi import Evaluator
import datetime

class DummyLogger():
    def __init__(self, path, params):
        self.stage = ''
        self.model = ''
        self.outfile = path + '{}_evaluation.log'.format(params.job_id)
    def write(self, string):
        now = datetime.datetime.now()
        with open(self.outfile, 'a') as f:
            output = '{} : {} {} {}\n'.format(now, self.stage, self.model, string)
            
            f.write(output)


if __name__ == '__main__':
    
    params = parse_game_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params.num_actions = 5 if params.limit_actions else 8
    obs_shape = (3, params.screen_height, params.screen_width)

    print('creating evaluators')
    test_evaluator = Evaluator(params, is_train=False)
    print('Test evaluator created')
    train_evaluator = Evaluator(params, is_train=True)
    print('Train evaluator created')


    if params.ego_model:
        actor_critic = EgoMap0_Policy(obs_shape[0], obs_shape, params)
    else:
        actor_critic = CNNPolicy(obs_shape[0], obs_shape, params)

    models_filepath = params.model_checkpoint
    dummy_logger = DummyLogger(models_filepath, params)
    directories = [f for f in os.listdir(models_filepath) if 'log' not in f]
    print(directories)
      
    j=0

    for directory in sorted(directories):
        for stage, flag, evaluator in [('test', False, test_evaluator), ('train', True, train_evaluator)]:
            dummy_logger.stage = stage
            dummy_logger.model = directory
            model_names = os.listdir(models_filepath + directory)  
            for name in sorted(model_names):
                
                path = models_filepath + directory + '/' + name    
                total_num_steps = int(name[11:21])
                print(name, total_num_steps)
                checkpoint = torch.load(path, map_location=lambda storage, loc: storage) 
                actor_critic.load_state_dict(checkpoint['model'])    
                actor_critic = actor_critic.to(device)
                with torch.no_grad():
                    evaluator.evaluate(actor_critic, 
                                       params, 
                                       dummy_logger, j, 
                                       total_num_steps, 
                                       params.eval_games,
                                       movie=False) 


    test_evaluator.cancel()
    train_evaluator.cancel()