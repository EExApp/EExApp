#!/usr/bin/env python3
"""
Example script demonstrating PPO training with integrated visualization.
This script shows how to use the enhanced ppo.py with visualization capabilities.
"""

import os
import sys
import torch
import numpy as np
from ppo import ppo_mcma, main
from env import OranEnv
from config import config

def run_ppo_with_visualization():
    """
    Run PPO training with integrated visualization.
    """
    print("Starting PPO training with integrated visualization...")
    print("=" * 60)
    
    # Set up environment function
    def env_fn():
        return OranEnv(
            num_slices=config.ENV['num_slices'],
            N_sf=config.ENV['N_sf'],
            user_num=config.ENV['user_num']
        )
    
    # Run PPO with visualization
    ppo_mcma(
        env_fn=env_fn,
        epochs=50,  # Reduced for demonstration
        steps_per_epoch=30,  # Reduced for demonstration
        save_dir="example_ppo_results",
        save_freq=10
    )
    
    print("Training completed! Check the 'example_ppo_results' directory for:")
    print("- Training plots in 'plots/' subdirectory")
    print("- Model checkpoints in 'models/' subdirectory")
    print("- Interactive dashboard: 'plots/training_dashboard.html'")

def run_quick_test():
    """
    Run a quick test to verify the integration works.
    """
    print("Running quick test of PPO with visualization...")
    print("=" * 60)
    
    # Set up environment function
    def env_fn():
        return OranEnv(
            num_slices=config.ENV['num_slices'],
            N_sf=config.ENV['N_sf'],
            user_num=config.ENV['user_num']
        )
    
    # Run PPO with minimal epochs for testing
    ppo_mcma(
        env_fn=env_fn,
        epochs=5,  # Very few epochs for quick test
        steps_per_epoch=10,  # Very few steps for quick test
        save_dir="quick_test_results",
        save_freq=2
    )
    
    print("Quick test completed! Check 'quick_test_results' directory.")

def check_visualization_files():
    """
    Check if visualization files were created successfully.
    """
    print("Checking for visualization files...")
    print("=" * 60)
    
    # Check for different result directories
    result_dirs = ["example_ppo_results", "quick_test_results", "ppo_training_results"]
    
    for result_dir in result_dirs:
        if os.path.exists(result_dir):
            print(f"\nFound results directory: {result_dir}")
            
            # Check plots directory
            plots_dir = os.path.join(result_dir, "plots")
            if os.path.exists(plots_dir):
                print(f"  ✓ Plots directory: {plots_dir}")
                
                # List plot files
                plot_files = [f for f in os.listdir(plots_dir) if f.endswith(('.png', '.html'))]
                if plot_files:
                    print(f"  ✓ Found {len(plot_files)} plot files:")
                    for file in plot_files[:5]:  # Show first 5 files
                        print(f"    - {file}")
                    if len(plot_files) > 5:
                        print(f"    ... and {len(plot_files) - 5} more")
                else:
                    print("  ✗ No plot files found")
            
            # Check models directory
            models_dir = os.path.join(result_dir, "models")
            if os.path.exists(models_dir):
                print(f"  ✓ Models directory: {models_dir}")
                
                # List model files
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                if model_files:
                    print(f"  ✓ Found {len(model_files)} model files:")
                    for file in model_files[:3]:  # Show first 3 files
                        print(f"    - {file}")
                    if len(model_files) > 3:
                        print(f"    ... and {len(model_files) - 3} more")
                else:
                    print("  ✗ No model files found")
        else:
            print(f"✗ Results directory not found: {result_dir}")

if __name__ == "__main__":
    print("PPO with Visualization Example")
    print("=" * 60)
    print("1. Run quick test (5 epochs)")
    print("2. Run full example (50 epochs)")
    print("3. Check visualization files")
    print("4. Run main function (command line args)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        run_quick_test()
    elif choice == "2":
        run_ppo_with_visualization()
    elif choice == "3":
        check_visualization_files()
    elif choice == "4":
        # Run the main function which handles command line arguments
        main()
    else:
        print("Invalid choice. Running quick test by default...")
        run_quick_test() 