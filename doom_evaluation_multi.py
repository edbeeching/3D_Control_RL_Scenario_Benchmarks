#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08/10/2018

@author: anonymous
"""

import numpy as np
from moviepy.editor import ImageSequenceClip

import torch
from arguments import parse_game_args
from models import CNNPolicy
from multi_env import MultiEnvsMPPipes



class Scorer():
    def __init__(self, num_envs, initial_obs, movie=True):
        self.best = [None, -100000] # obs, and best reward
        self.worst = [None, 100000] # obs, and worse reward
        self.trajectories = {}
        self.total_rewards = []
        self.total_times = []
        self.num_envs = num_envs
        self.movie = movie
        if self.movie:
            initial_obs = initial_obs.astype(np.uint8)           
        else:
            initial_obs = [None]*initial_obs.shape[0]
        
        for i in range(num_envs):
            self.trajectories[i] = [[initial_obs[i]], []]
            
    def update(self, obs, rewards, dones):   
        obs = obs.astype(np.uint8)   
        if self.movie:
            obs = obs.astype(np.uint8)           
        else:
            obs = [None]*obs.shape[0]
        
        
        for i in range(self.num_envs):
            if dones[i]:
                self.trajectories[i][1].append(rewards[i])
                accumulated_reward = sum(self.trajectories[i][1])
                self.total_rewards.append(accumulated_reward)
                self.total_times.append(len(self.trajectories[i][1]))
                
                if accumulated_reward > self.best[1]:
                    self.best[0] = self.trajectories[i][0]
                    self.best[1] = accumulated_reward
                    
                if accumulated_reward < self.worst[1]:
                    self.worst[0] = self.trajectories[i][0]
                    self.worst[1] = accumulated_reward   
             
                self.trajectories[i] = [[obs[i]], [0.0]]
  
            else:
                self.trajectories[i][0].append(obs[i])
                self.trajectories[i][1].append(rewards[i])
                
                
    def clear(self):
        self.trajectories = None
        
         
class Evaluator():
    def __init__(self, params, is_train=False):
        self.num_envs = params.num_environments
        self.envs = MultiEnvsMPPipes(params.simulator, self.num_envs, 1, params, is_train=is_train)  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        self.states = torch.zeros(self.num_envs, params.hidden_size).to(self.device)
         
    def cancel(self):
        self.envs.cancel()
    
    def evaluate(self, model,  params,logger,  step, train_iters, num_games=10, movie=True):
        model.eval()
        
        games_played = 0
        obs = self.envs.reset()
        # add obs to scorer
        scorer = Scorer(self.num_envs, obs, movie=movie)
        obs = torch.from_numpy(obs).float().to(self.device)
        masks = torch.ones(self.num_envs, 1).to(self.device)
  
        self.states.zero_().detach()
   
        while games_played < num_games:
            _, actions, _, self.states= model.act(
                obs, self.states.detach(),masks,deterministic=not params.stoc_evals)        
    
            cpu_actions = actions.squeeze(1).cpu().numpy()
            obs, reward, done, info = self.envs.step(cpu_actions)
            # add obs, reward,  to scorer
            scorer.update(obs, reward, done)
    
            games_played += done.count(True) # done is true at end of a turn

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            masks = masks.to(self.device)
            obs = torch.from_numpy(obs).float().to(self.device)
    
        model.train()
        
         # it is possible that this is larger that the total num games
        accumulated_rewards = sum(scorer.total_rewards[:num_games])
        best_obs, best_reward = scorer.best
        worst_obs, worst_reward = scorer.worst
        reward_list = scorer.total_rewards[:num_games]
        time_list = scorer.total_times[:num_games]
        scorer.clear()
        
        if params.use_visdom:
            logger.vis_iters.append(train_iters)
            logger.vis_scores.append(accumulated_rewards / num_games)
            logger.update_plot(train_iters)
            
        if movie:
            write_movie(params, logger, best_obs, step, best_reward)
            write_movie(params, logger, worst_obs, step+1, worst_reward, best_agent=False)    
            
        logger.write('Step: {:0004}, Iter: {:000000008} Eval mean reward: {:0003.3f}'.format(step, train_iters, accumulated_rewards / num_games))
        logger.write('Step: {:0004}, Game rewards: {}, Game times: {}'.format(step, reward_list, time_list))         

def write_movie(params, logger,  observations, step, score, best_agent=True):    
    observations = [o.transpose(1,2,0) for o in observations]
    clip = ImageSequenceClip(observations, fps=int(30/params.frame_skip))
    output_dir = logger.get_eval_output()
    clip.write_videofile('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100))  
    if params.use_visdom:
        logger.add_video('{}eval{:0004}_{:00005.0f}.mp4'.format(output_dir, step, score*100), best_agent=best_agent)
    

  
if __name__ == '__main__':
    params = parse_game_args()    
    params.norm_obs = False
    params.num_stack = 1
    params.recurrent_policy = True
    params.num_environments = 16
    params.scenario = 'scenario_3_item0.cfg'
    
    envs = MultiEnvsMPPipes(params.simulator, 1, 1, params)
    obs_shape = envs.obs_shape
    obs_shape = (obs_shape[0] * params.num_stack, *obs_shape[1:])    
    model = CNNPolicy(obs_shape[0], obs_shape, params)

    with torch.no_grad():
        eval_model_multi(model,  params, 0, 0, 0, num_games=1000)    
        
        
  
