#!/usr/bin/env python3
"""
Script to plot moving averages of epoch return and policy loss from SGRPO training.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_training_metrics(metrics_file):
    """Load training metrics from JSON file."""
    with open(metrics_file, 'r') as f:
        data = json.load(f)
    return data

def calculate_moving_average(data, window_size=5):
    """Calculate moving average with specified window size."""
    if len(data) < window_size:
        return data  # Return original data if not enough points
    
    moving_avg = []
    for i in range(len(data)):
        if i < window_size - 1:
            # For early points, use available data
            moving_avg.append(np.mean(data[:i+1]))
        else:
            # Use full window
            moving_avg.append(np.mean(data[i-window_size+1:i+1]))
    
    return moving_avg

def plot_epoch_return_moving_average(metrics_data, save_path="sgrpo_training_plots"):
    """Plot epoch return with moving average."""
    epochs = metrics_data['epoch']
    ep_ret = metrics_data['ep_ret']
    ma_7 = calculate_moving_average(ep_ret, window_size=7)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ma_7, 'o-', color='darkblue', linewidth=3, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Epoch Return', fontsize=12)
    plt.title('SGRPO Training: Epoch Return', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(save_path) / "epoch_return_moving_average.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Epoch return moving average plot saved to: {output_path}")
    plt.show()

def plot_policy_loss_moving_average(metrics_data, save_path="sgrpo_training_plots"):
    """Plot policy loss with moving average."""
    epochs = metrics_data['epoch']
    pi_loss = metrics_data['pi_loss']
    ma_7 = calculate_moving_average(pi_loss, window_size=7)
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ma_7, 'o-', color='darkred', linewidth=3, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Policy Loss', fontsize=12)
    plt.title('SGRPO Training: Policy Loss (Moving Average, window=7)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(save_path) / "policy_loss_moving_average.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Policy loss moving average plot saved to: {output_path}")
    plt.show()

def plot_combined_moving_averages(metrics_data, save_path="sgrpo_training_plots"):
    """Plot both metrics in a single figure with subplots."""
    epochs = metrics_data['epoch']
    ep_ret = metrics_data['ep_ret']
    pi_loss = metrics_data['pi_loss']
    ma_ret_7 = calculate_moving_average(ep_ret, window_size=7)
    ma_loss_7 = calculate_moving_average(pi_loss, window_size=7)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    ax1.plot(epochs, ma_ret_7, 'o-', color='darkblue', linewidth=3, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Epoch Return', fontsize=12)
    ax1.set_title('SGRPO Training: Epoch Return (Moving Average, window=7)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax2.plot(epochs, ma_loss_7, 'o-', color='darkred', linewidth=3, markersize=6)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Policy Loss', fontsize=12)
    ax2.set_title('SGRPO Training: Policy Loss (Moving Average, window=7)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(save_path) / "combined_moving_averages.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Combined moving averages plot saved to: {output_path}")
    plt.show()

def plot_loss_kl_moving_average(metrics_data, save_path="sgrpo_training_plots"):
    epochs = metrics_data['epoch']
    pi_loss = metrics_data['pi_loss']
    kl_div = metrics_data['kl_div']
    ma_loss = calculate_moving_average(pi_loss, window_size=15)  # Smoother loss
    ma_kl = calculate_moving_average(kl_div, window_size=7)      # KL as before
    plt.figure(figsize=(12, 8))
    plt.plot(epochs, ma_loss, 'o-', color='darkred', linewidth=3, markersize=6)
    plt.plot(epochs, ma_kl, 'o-', color='navy', linewidth=3, markersize=6)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.title('SGRPO Training: Policy Loss and KL Divergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path = Path(save_path) / "loss_kl_moving_average.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Policy loss and KL divergence plot saved to: {output_path}")
    plt.show()

def main():
    """Main function to generate all moving average plots."""
    # Load training metrics
    metrics_file = "sgrpo_training_plots/sgrpo_training_metrics.json"
    
    try:
        metrics_data = load_training_metrics(metrics_file)
        print(f"Loaded training metrics for {len(metrics_data['epoch'])} epochs")
        
        # Generate individual plots
        plot_epoch_return_moving_average(metrics_data)
        plot_policy_loss_moving_average(metrics_data)
        
        # Generate combined plot
        plot_combined_moving_averages(metrics_data)
        plot_loss_kl_moving_average(metrics_data)
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"Epoch Return - Mean: {np.mean(metrics_data['ep_ret']):.2f}, Std: {np.std(metrics_data['ep_ret']):.2f}")
        print(f"Policy Loss - Mean: {np.mean(metrics_data['pi_loss']):.3f}, Std: {np.std(metrics_data['pi_loss']):.3f}")
        print(f"KL Divergence - Mean: {np.mean(metrics_data['kl_div']):.3f}, Std: {np.std(metrics_data['kl_div']):.3f}")
        
    except FileNotFoundError:
        print(f"Error: Could not find metrics file: {metrics_file}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 