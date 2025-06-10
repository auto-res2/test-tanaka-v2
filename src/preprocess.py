#!/usr/bin/env python3
"""
Data preprocessing utilities for RDEIL experiments.
Handles expert dataset generation, noise injection, and data transformations.
"""

import gymnasium as gym
import numpy as np
import torch

def generate_expert_dataset(env, num_samples=1000):
    """
    Generates a dataset of (state, action) pairs using an expert policy.
    For demonstration, we use the env.action_space.sample() as a pseudo-expert.
    """
    dataset = []
    obs, _ = env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()  
        dataset.append((obs, action))
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    return dataset

def inject_noise(actions, noise_std):
    """
    Inject Gaussian noise into a numpy array of actions.
    """
    actions = np.array(actions)
    noise = np.random.randn(*actions.shape) * noise_std
    return actions + noise

def add_state_noise(states, noise_std):
    """
    Add Gaussian noise into state observations to mimic a domain shift.
    """
    noise = np.random.randn(*states.shape) * noise_std
    return states + noise

def adaptive_noise(state_batch, base_std=0.1, critical_std=0.3, threshold=1.0):
    """
    For each state sample, decide via a heuristic (here, the L2 norm of the state)
    whether it is a safety-critical state (i.e., high norm) and require high noise.
    Returns a tensor of noise standard deviations.
    """
    state_np = state_batch.detach().cpu().numpy()
    noise_std_array = np.array([critical_std if np.linalg.norm(s) > threshold else base_std for s in state_np])
    return torch.tensor(noise_std_array, dtype=torch.float32).unsqueeze(1)
