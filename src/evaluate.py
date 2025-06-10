#!/usr/bin/env python3
"""
Evaluation module for RDEIL experiments.
Contains policy evaluation and performance measurement utilities.
"""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_policy(policy_model, env, noise_std=0.0, num_episodes=5):
    """
    Evaluate a policy in the given environment with optional state noise.
    Returns average cumulative reward over episodes.
    """
    total_reward = 0.0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            noisy_obs = obs + np.random.randn(*np.array(obs).shape) * noise_std
            state_tensor = torch.tensor(noisy_obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy_model(state_tensor).detach().cpu().numpy()[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        total_reward += episode_reward
    return total_reward / num_episodes

def plot_training_curves(baseline_losses, rdeil_losses, recon_errors, save_path):
    """Plot training loss curves for baseline vs RDEIL."""
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(baseline_losses) + 1)
    plt.plot(epochs, baseline_losses, label="Baseline Loss", marker='o')
    plt.plot(epochs, rdeil_losses, label="RDEIL Loss", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_reconstruction_error(recon_errors, save_path):
    """Plot reconstruction error over training epochs."""
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(recon_errors) + 1)
    plt.plot(epochs, recon_errors, label="Reconstruction Error", marker='^', color='purple')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reconstruction (Denoising) Error over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_domain_shift_performance(noise_values, baseline_rewards, rdeil_rewards, save_path):
    """Plot policy performance under domain shift."""
    plt.figure(figsize=(8, 6))
    plt.plot(noise_values, baseline_rewards, label="Baseline Policy", marker='o')
    plt.plot(noise_values, rdeil_rewards, label="RDEIL Policy", marker='s')
    plt.xlabel("State Noise STD")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Policy Reward under Domain Shift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def plot_adaptive_vs_fixed(fixed_losses, adaptive_losses, save_path):
    """Plot comparison of adaptive vs fixed noise schedules."""
    plt.figure(figsize=(8, 6))
    epochs = np.arange(1, len(fixed_losses) + 1)
    plt.plot(epochs, fixed_losses, label="RDEIL Fixed Noise", marker='o')
    plt.plot(epochs, adaptive_losses, label="RDEIL Adaptive Noise", marker='s')
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Training Loss Curves (Fixed vs Adaptive Noise)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

def save_results_summary(results, save_path):
    """Save experiment results summary to text file."""
    with open(save_path, 'w') as f:
        f.write("RDEIL Experiment Results Summary\n")
        f.write("=" * 40 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
