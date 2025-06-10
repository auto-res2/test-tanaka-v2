#!/usr/bin/env python3
"""
Training module for RDEIL experiments.
Contains network definitions and training utilities.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNN(nn.Module):
    """Neural network for policy learning."""
    def __init__(self, input_dim, output_dim):
        super(PolicyNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

class DenoisingModule(nn.Module):
    """Denoising module for RDEIL diffusion process."""
    def __init__(self, input_dim, output_dim):
        super(DenoisingModule, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, noisy_action):
        return self.fc(noisy_action)

def train_baseline_policy(policy, states, actions, num_epochs=20, batch_size=64, lr=1e-3):
    """Train baseline policy using standard behavioral cloning."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    loss_history = []
    
    for epoch in range(num_epochs):
        permutation = np.random.permutation(len(states))
        epoch_losses = []
        
        for i in range(0, len(states), batch_size):
            indices = permutation[i:i+batch_size]
            state_batch = torch.tensor(states[indices])
            action_batch = torch.tensor(actions[indices]).float()
            
            optimizer.zero_grad()
            pred_action = policy(state_batch)
            loss = mse_loss(pred_action, action_batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
    
    return loss_history

def train_rdeil_policy(policy, denoiser, states, clean_actions, noisy_actions, 
                      num_epochs=20, batch_size=64, lr=1e-3, lambda_diff=1.0):
    """Train RDEIL policy with combined BC and denoising loss."""
    optimizer = optim.Adam(list(policy.parameters()) + list(denoiser.parameters()), lr=lr)
    mse_loss = nn.MSELoss()
    loss_history = []
    recon_history = []
    
    for epoch in range(num_epochs):
        permutation = np.random.permutation(len(states))
        epoch_losses = []
        epoch_recon = []
        
        for i in range(0, len(states), batch_size):
            indices = permutation[i:i+batch_size]
            state_batch = torch.tensor(states[indices])
            clean_action_batch = torch.tensor(clean_actions[indices]).float()
            noisy_action_batch = torch.tensor(noisy_actions[indices]).float()
            
            optimizer.zero_grad()
            pred_action = policy(state_batch)
            bc_loss = mse_loss(pred_action, noisy_action_batch)
            denoised_action = denoiser(pred_action)
            diffusion_loss = mse_loss(denoised_action, clean_action_batch)
            combined_loss = bc_loss + lambda_diff * diffusion_loss
            combined_loss.backward()
            optimizer.step()
            
            epoch_losses.append(combined_loss.item())
            epoch_recon.append(diffusion_loss.item())
        
        avg_loss = np.mean(epoch_losses)
        avg_recon = np.mean(epoch_recon)
        loss_history.append(avg_loss)
        recon_history.append(avg_recon)
    
    return loss_history, recon_history
