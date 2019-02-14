#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 11:18:31 2018

@author: anonymous

This code builds upon work found here:
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr
    


"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from logger import Logger
from multi_env import MultiEnvsMPPipes
from arguments import parse_game_args
from rollout_buffer import RolloutStorage
from models import CNNPolicy
from doom_evaluation_multi import Evaluator

def train():
    # define params
    params = parse_game_args()
    logger = Logger(params)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_updates = int(params.num_frames) // params.num_steps // params.num_environments

    # environments

    envs = MultiEnvsMPPipes(params.simulator, params.num_environments, 1, params)

    obs_shape = envs.obs_shape
    obs_shape = (obs_shape[0] * params.num_stack, *obs_shape[1:])
    
    evaluator = Evaluator(params) 
    print('creating model')
    actor_critic = CNNPolicy(obs_shape[0], obs_shape, params).to(device) 
    print('model created')
    start_j = 0

    if params.reload_model:
        checkpoint_idx = params.reload_model.split(',')[1]
        checkpoint_filename = '{}models/checkpoint_{}.pth.tar'.format(params.output_dir, checkpoint_idx)        
        assert os.path.isfile(checkpoint_filename), 'The model could not be found {}'.format(checkpoint_filename)
        logger.write('Loading model{}'.format( checkpoint_filename))
        
        if device == 'cuda': # The checkpoint will try to load onto the GPU storage unless specified
            checkpoint = torch.load(checkpoint_filename)
        else:
            checkpoint = torch.load(checkpoint_filename, map_location=lambda storage, loc: storage)
        actor_critic.load_state_dict(checkpoint['model'])        
        
        start_j = (int(checkpoint_idx) // params.num_steps // params.num_environments) + 1

    print('creating optimizer')
    optimizer = optim.RMSprop([p for p in actor_critic.parameters() if p.requires_grad], 
                              params.learning_rate, 
                              eps=params.eps, 
                              alpha=params.alpha, 
                              momentum=params.momentum)
        
    if params.reload_model:
        optimizer.load_state_dict(checkpoint['optimizer'])

    
    rollouts = RolloutStorage(params.num_steps, 
                              params.num_environments, 
                              obs_shape, 
                              actor_critic.state_size, 
                              params)
    
    current_obs = torch.zeros(params.num_environments, *obs_shape)

    # For Frame stacking
    def update_current_obs(obs):
        shape_dim0 = envs.obs_shape[0]
        obs = torch.from_numpy(obs).float()
        if params.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs
        
    print('getting first obs')
    obs = envs.reset()
    print('update current obs')
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)
      
    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([params.num_environments, 1])
    final_rewards = torch.zeros([params.num_environments, 1])
    
    current_obs = current_obs.to(device)
    rollouts.set_device(device)
        
    print('Starting training loop')
    start = time.time()
    print(num_updates)
     
    for j in range(start_j, num_updates):
        # STARTING no grad scope
        with torch.no_grad():
            
            if j % params.eval_freq == 0 and not params.skip_eval:
                print('Evaluating model')
                if params.simulator == 'doom':
                    actor_critic.eval()
                    total_num_steps = (j + 1) * params.num_environments * params.num_steps    
                    #eval_model(actor_critic, params, logger, j, total_num_steps, params.eval_games) 
                    evaluator.evaluate(actor_critic, params, logger, j, total_num_steps, params.eval_games) 
                    actor_critic.train()
                    
            # =============================================================================
            # Take steps in the environment   
            # =============================================================================
            for step in range(params.num_steps):
                # Sample actions
                value, action, action_log_prob, states = actor_critic.act(
                        rollouts.observations[step],
                        rollouts.states[step],
                        rollouts.masks[step])
                
                cpu_actions = action.squeeze(1).cpu().numpy()
    
                # Obser reward and next obs
                obs, reward, done, info = envs.step(cpu_actions)
            
                reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
                episode_rewards += reward
    
                # If done then create masks to clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks
    
                masks = masks.to(device)
    
                if current_obs.dim() == 4:
                    current_obs *= masks.unsqueeze(2).unsqueeze(2)
                else:
                    current_obs *= masks
    
                update_current_obs(obs)
                
                rollouts.insert(step, current_obs, states, action, 
                                action_log_prob, value, reward, masks)
    
            # =============================================================================
            # Compute discounted returns, re-step through the environment
            # =============================================================================           
            next_value = actor_critic(rollouts.observations[-1],
                                      rollouts.states[-1],
                                      rollouts.masks[-1])[0]
    
            rollouts.compute_returns(next_value, params.use_gae, params.gamma, params.tau)
        
        # FINISHED no grad scope
        model_output = actor_critic.evaluate_actions(rollouts.observations[:-1].view(-1, *obs_shape),
                                                     rollouts.states[0].view(-1, actor_critic.state_size),
                                                     rollouts.masks[:-1].view(-1, 1),
                                                     rollouts.actions.view(-1, 1))
        
        values, action_log_probs, dist_entropy, states = model_output

        values = values.view(params.num_steps, params.num_environments, 1)
        action_log_probs = action_log_probs.view(params.num_steps, params.num_environments, 1)
        advantages =rollouts.returns[:-1] - values
        
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        optimizer.zero_grad()
       
        loss = value_loss * params.value_loss_coef + action_loss - dist_entropy * params.entropy_coef
        loss.backward()
        nn.utils.clip_grad_norm(actor_critic.parameters(), params.max_grad_norm)
         
        optimizer.step()
        rollouts.after_update()

        if j % params.model_save_rate == 0:
            total_num_steps = (j + 1) * params.num_environments * params.num_steps
            checkpoint = {'step': step,
              'params': params,
              'model': actor_critic.state_dict(),
              'optimizer': optimizer.state_dict()}
        
            filepath = logger.output_dir + 'models/'
        
            torch.save(checkpoint, '{}checkpoint_{:00000000010}.pth.tar'.format(filepath, total_num_steps))

        if j % params.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * params.num_environments * params.num_steps
            save_num_steps = (start_j) * params.num_environments * params.num_steps
            logger.write("Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int((total_num_steps - save_num_steps) / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.item(),
                       value_loss.item(), action_loss.item()))
            
            
    evaluator.cancel()
    envs.cancel()


if __name__ == "__main__":
    train()


