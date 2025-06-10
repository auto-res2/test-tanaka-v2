#!/usr/bin/env python3
"""
Main experiment script for RDEIL (Robust Diffusion-Enhanced Imitation Learning).
Implements three experiments: controlled noise injection, domain shift generalization, 
and adaptive vs fixed noise schedules.
"""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

from preprocess import generate_expert_dataset, inject_noise, adaptive_noise
from train import PolicyNN, DenoisingModule, train_baseline_policy, train_rdeil_policy
from evaluate import (evaluate_policy, plot_training_curves, plot_reconstruction_error,
                     plot_domain_shift_performance, plot_adaptive_vs_fixed, save_results_summary)

torch.manual_seed(42)
np.random.seed(42)

def experiment1_controlled_noise(num_epochs=20, batch_size=64, verbose=True):
    """Experiment 1: Controlled Noise Injection Test"""
    print("\nStarting Experiment 1: Controlled Noise Injection Test")
    env = gym.make('Pendulum-v1')
    dataset = generate_expert_dataset(env, num_samples=2000)
    states, expert_actions = zip(*dataset)
    states = np.array(states, dtype=np.float32)
    expert_actions = np.array(expert_actions, dtype=np.float32)
    
    input_dim = states.shape[1]
    output_dim = expert_actions.shape[1]
    
    noise_std_schedule = np.linspace(0.0, 0.5, num_epochs)
    
    baseline_policy = PolicyNN(input_dim, output_dim)
    rdeil_policy = PolicyNN(input_dim, output_dim)
    denoiser = DenoisingModule(output_dim, output_dim)
    
    baseline_loss_history = []
    rdeil_loss_history = []
    reconstruction_error_history = []
    
    for epoch in range(num_epochs):
        noise_std = noise_std_schedule[epoch]
        noisy_actions = inject_noise(expert_actions, noise_std)
        
        baseline_losses = train_baseline_policy(baseline_policy, states, noisy_actions, 
                                              num_epochs=1, batch_size=batch_size)
        baseline_loss_history.extend(baseline_losses)
        
        rdeil_losses, recon_errors = train_rdeil_policy(rdeil_policy, denoiser, states, 
                                                       expert_actions, noisy_actions,
                                                       num_epochs=1, batch_size=batch_size)
        rdeil_loss_history.extend(rdeil_losses)
        reconstruction_error_history.extend(recon_errors)
        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}: Baseline Loss = {baseline_losses[0]:.4f}, "
                  f"RDEIL Combined Loss = {rdeil_losses[0]:.4f}, Reconstruction Error = {recon_errors[0]:.4f}")
    
    images_dir = ".research/iteration1/images"
    os.makedirs(images_dir, exist_ok=True)
    
    plot_training_curves(baseline_loss_history, rdeil_loss_history, reconstruction_error_history,
                        os.path.join(images_dir, "training_loss_experiment1.pdf"))
    plot_reconstruction_error(reconstruction_error_history,
                            os.path.join(images_dir, "reconstruction_error_experiment1.pdf"))
    
    print("Experiment 1 completed. Plots saved.")
    return baseline_policy, rdeil_policy

def experiment2_domain_shift(baseline_policy, rdeil_policy, verbose=True):
    """Experiment 2: Generalization Under Domain Shift"""
    print("\nStarting Experiment 2: Generalization Under Domain Shift")
    env = gym.make('Pendulum-v1')
    noise_values = [0.0, 0.1, 0.2, 0.3]
    baseline_rewards = []
    rdeil_rewards = []
    
    for noise in noise_values:
        b_reward = evaluate_policy(baseline_policy, env, noise_std=noise, num_episodes=5)
        r_reward = evaluate_policy(rdeil_policy, env, noise_std=noise, num_episodes=5)
        baseline_rewards.append(b_reward)
        rdeil_rewards.append(r_reward)
        if verbose:
            print(f"State Noise STD {noise:.2f}: Baseline Reward = {b_reward:.2f}, RDEIL Reward = {r_reward:.2f}")
    
    images_dir = ".research/iteration1/images"
    plot_domain_shift_performance(noise_values, baseline_rewards, rdeil_rewards,
                                os.path.join(images_dir, "policy_performance_domain_shift.pdf"))
    
    print("Experiment 2 completed. Plot saved.")

def experiment3_adaptive_vs_fixed(num_epochs=20, batch_size=64, verbose=True):
    """Experiment 3: Adaptive Noise Schedule versus Fixed Noise Schedule"""
    print("\nStarting Experiment 3: Adaptive Noise Schedule versus Fixed Noise Schedule")
    env = gym.make('Pendulum-v1')
    dataset = generate_expert_dataset(env, num_samples=2000)
    states, expert_actions = zip(*dataset)
    states = np.array(states, dtype=np.float32)
    expert_actions = np.array(expert_actions, dtype=np.float32)
    input_dim = states.shape[1]
    output_dim = expert_actions.shape[1]
    
    fixed_noise_std = 0.2
    
    rdeil_fixed = PolicyNN(input_dim, output_dim)
    rdeil_adaptive = PolicyNN(input_dim, output_dim)
    denoiser_fixed = DenoisingModule(output_dim, output_dim)
    denoiser_adaptive = DenoisingModule(output_dim, output_dim)
    
    fixed_loss_history = []
    adaptive_loss_history = []
    
    for epoch in range(num_epochs):
        noisy_actions_fixed = inject_noise(expert_actions, fixed_noise_std)
        fixed_losses, _ = train_rdeil_policy(rdeil_fixed, denoiser_fixed, states,
                                           expert_actions, noisy_actions_fixed,
                                           num_epochs=1, batch_size=batch_size)
        fixed_loss_history.extend(fixed_losses)
        
        state_tensor = torch.tensor(states)
        adaptive_std = adaptive_noise(state_tensor, base_std=0.1, critical_std=0.3, threshold=1.0)
        noisy_actions_adaptive = expert_actions + np.random.randn(*expert_actions.shape) * adaptive_std.numpy()
        adaptive_losses, _ = train_rdeil_policy(rdeil_adaptive, denoiser_adaptive, states,
                                              expert_actions, noisy_actions_adaptive,
                                              num_epochs=1, batch_size=batch_size)
        adaptive_loss_history.extend(adaptive_losses)
        
        if verbose:
            print(f"[Epoch {epoch+1}/{num_epochs}] Fixed Loss = {fixed_losses[0]:.4f}, "
                  f"Adaptive Loss = {adaptive_losses[0]:.4f}")
    
    images_dir = ".research/iteration1/images"
    plot_adaptive_vs_fixed(fixed_loss_history, adaptive_loss_history,
                          os.path.join(images_dir, "training_loss_rdeil_adaptive_vs_fixed.pdf"))
    
    print("Experiment 3 completed. Plot saved.")

def test_code():
    """Run a quick test for the experiments with reduced parameters."""
    print("Running quick test on all experiments.\n")
    start_time = time.time()
    
    baseline_policy, rdeil_policy = experiment1_controlled_noise(num_epochs=5, batch_size=64, verbose=True)
    
    experiment2_domain_shift(baseline_policy, rdeil_policy, verbose=True)
    
    experiment3_adaptive_vs_fixed(num_epochs=5, batch_size=64, verbose=True)
    
    end_time = time.time()
    print(f"\nTest execution completed in {end_time - start_time:.2f} seconds.\n")

def main():
    """Main function to run all experiments."""
    print("RDEIL (Robust Diffusion-Enhanced Imitation Learning) Experiments")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_code()
        return
    
    start_time = time.time()
    
    baseline_policy, rdeil_policy = experiment1_controlled_noise(num_epochs=20, batch_size=64)
    experiment2_domain_shift(baseline_policy, rdeil_policy)
    experiment3_adaptive_vs_fixed(num_epochs=20, batch_size=64)
    
    end_time = time.time()
    
    results = {
        "Total Runtime (seconds)": f"{end_time - start_time:.2f}",
        "Experiments Completed": 3,
        "Plots Generated": 4,
        "Status": "completed"
    }
    
    images_dir = ".research/iteration1/images"
    save_results_summary(results, os.path.join(images_dir, "experiment_summary.txt"))
    
    print(f"\nAll experiments completed successfully in {end_time - start_time:.2f} seconds!")
    print("Results and plots saved to .research/iteration1/images/")
    
    status_enum = "stopped"
    print(f"Status: {status_enum}")

if __name__ == "__main__":
    main()
