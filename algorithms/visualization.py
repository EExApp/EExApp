"""
Comprehensive visualization tools for PPO training in O-RAN control system.
Tracks all key metrics: training progress, network performance, energy efficiency, and action analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ORANVisualizer:
    """
    Comprehensive visualizer for O-RAN PPO training with dual-actor system.
    """
    
    def __init__(self, save_dir: str = "training_plots", config=None):
        self.save_dir = save_dir
        self.config = config
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metric tracking
        self.metrics = {
            # Training metrics
            'epoch': [], 'ep_ret': [], 'ep_len': [], 'pi_loss': [], 'v_loss': [],
            'kl_div': [], 'entropy': [], 'clip_frac': [], 'explained_var': [],
            
            # Reward decomposition
            'ep_ee_reward': [], 'ep_ns_reward': [], 'ep_total_reward': [],
            
            # Network performance
            'throughput_embb': [], 'throughput_urllc': [], 'throughput_mmtc': [],
            'delay_embb': [], 'delay_urllc': [], 'delay_mmtc': [],
            'qos_violations': [], 'resource_utilization': [],
            
            # Energy efficiency
            'energy_consumption': [], 'sleep_efficiency': [], 'energy_per_bit': [],
            
            # Actions
            'ee_actions': [], 'ns_actions': [], 'constraint_satisfaction': [],
            
            # Multi-objective
            'pareto_efficiency': [], 'objective_conflicts': [], 'fairness_index': []
        }
        
        # Real-time monitoring
        self.monitor_interval = 10
        self.last_plot_epoch = 0
        
    def update_metrics(self, epoch: int, training_metrics: Dict, 
                      network_metrics: Dict, action_metrics: Dict):
        """Update all metrics with new data."""
        try:
            self.metrics['epoch'].append(epoch)
            
            # Training metrics
            for key in ['ep_ret', 'ep_len', 'pi_loss', 'v_loss', 'kl_div', 
                       'entropy', 'clip_frac', 'explained_var']:
                if key in training_metrics and training_metrics[key] is not None:
                    self.metrics[key].append(training_metrics[key])
            
            # Reward decomposition
            for key in ['ep_ee_reward', 'ep_ns_reward', 'ep_total_reward']:
                if key in training_metrics and training_metrics[key] is not None:
                    self.metrics[key].append(training_metrics[key])
            
            # Network performance
            for key in ['throughput_embb', 'throughput_urllc', 'throughput_mmtc',
                       'delay_embb', 'delay_urllc', 'delay_mmtc', 'qos_violations',
                       'resource_utilization']:
                if key in network_metrics and network_metrics[key] is not None:
                    self.metrics[key].append(network_metrics[key])
            
            # Energy efficiency
            for key in ['energy_consumption', 'sleep_efficiency', 'energy_per_bit']:
                if key in network_metrics and network_metrics[key] is not None:
                    self.metrics[key].append(network_metrics[key])
            
            # Actions
            if 'ee_actions' in action_metrics and action_metrics['ee_actions'] is not None:
                self.metrics['ee_actions'].append(action_metrics['ee_actions'])
            if 'ns_actions' in action_metrics and action_metrics['ns_actions'] is not None:
                self.metrics['ns_actions'].append(action_metrics['ns_actions'])
            if 'constraint_satisfaction' in action_metrics and action_metrics['constraint_satisfaction'] is not None:
                self.metrics['constraint_satisfaction'].append(action_metrics['constraint_satisfaction'])
            
            # Multi-objective metrics
            for key in ['pareto_efficiency', 'objective_conflicts', 'fairness_index']:
                if key in training_metrics and training_metrics[key] is not None:
                    self.metrics[key].append(training_metrics[key])
                    
        except Exception as e:
            print(f"Warning: Error updating metrics for epoch {epoch}: {e}")
            # Continue with training even if visualization fails
    
    def plot_training_progress(self, save: bool = True, show: bool = False):
        """Plot core training progress metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PPO Training Progress - Core Metrics', fontsize=16)
        
        # Episode returns
        if self.metrics['ep_ret']:
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['ep_ret'], 'b-', linewidth=2)
            axes[0, 0].set_title('Episode Returns')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Add moving average
            if len(self.metrics['ep_ret']) > 10:
                window = min(10, len(self.metrics['ep_ret']) // 10)
                ma = pd.Series(self.metrics['ep_ret']).rolling(window=window).mean()
                axes[0, 0].plot(self.metrics['epoch'], ma, 'r--', linewidth=2, 
                               label=f'MA({window})')
                axes[0, 0].legend()
        
        # Losses
        if self.metrics['pi_loss']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['pi_loss'], 
                           'g-', linewidth=2, label='Policy Loss')
        if self.metrics['v_loss']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['v_loss'], 
                           'r-', linewidth=2, label='Value Loss')
        axes[0, 1].set_title('Training Losses')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # KL divergence
        if self.metrics['kl_div']:
            axes[0, 2].plot(self.metrics['epoch'], self.metrics['kl_div'], 'm-', linewidth=2)
            axes[0, 2].axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Target KL')
            axes[0, 2].set_title('KL Divergence')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('KL Divergence')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # Reward components
        if self.metrics['ep_ee_reward'] and self.metrics['ep_ns_reward']:
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['ep_ee_reward'], 
                           'c-', linewidth=2, label='Energy Efficiency')
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['ep_ns_reward'], 
                           'orange', linewidth=2, label='Network Slicing')
            axes[1, 0].set_title('Reward Components')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        if self.metrics['entropy']:
            axes[1, 1].plot(self.metrics['epoch'], self.metrics['entropy'], 
                           'purple', linewidth=2)
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Clipping fraction
        if self.metrics['clip_frac']:
            axes[1, 2].plot(self.metrics['epoch'], self.metrics['clip_frac'], 
                           'brown', linewidth=2)
            axes[1, 2].set_title('Clipping Fraction')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Clip Fraction')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'training_progress.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_network_performance(self, save: bool = True, show: bool = False):
        """Plot network performance metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Network Performance Metrics', fontsize=16)
        
        # Throughput per slice
        if self.metrics['throughput_embb']:
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['throughput_embb'], 
                           'blue', linewidth=2, label='eMBB')
        if self.metrics['throughput_urllc']:
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['throughput_urllc'], 
                           'red', linewidth=2, label='URLLC')
        if self.metrics['throughput_mmtc']:
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['throughput_mmtc'], 
                           'green', linewidth=2, label='mMTC')
        axes[0, 0].set_title('Throughput per Slice')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Throughput (kbps)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Delay per slice
        if self.metrics['delay_embb']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['delay_embb'], 
                           'blue', linewidth=2, label='eMBB')
        if self.metrics['delay_urllc']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['delay_urllc'], 
                           'red', linewidth=2, label='URLLC')
        if self.metrics['delay_mmtc']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['delay_mmtc'], 
                           'green', linewidth=2, label='mMTC')
        axes[0, 1].set_title('Delay per Slice')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Delay (ms)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # QoS violations
        if self.metrics['qos_violations']:
            axes[0, 2].plot(self.metrics['epoch'], self.metrics['qos_violations'], 
                           'red', linewidth=2)
            axes[0, 2].set_title('QoS Violations')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Number of Violations')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Resource utilization
        if self.metrics['resource_utilization']:
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['resource_utilization'], 
                           'orange', linewidth=2)
            axes[1, 0].set_title('Resource Utilization')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Utilization (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Energy consumption
        if self.metrics['energy_consumption']:
            axes[1, 1].plot(self.metrics['epoch'], self.metrics['energy_consumption'], 
                           'purple', linewidth=2)
            axes[1, 1].set_title('Energy Consumption')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Energy (J)')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Energy per bit
        if self.metrics['energy_per_bit']:
            axes[1, 2].plot(self.metrics['epoch'], self.metrics['energy_per_bit'], 
                           'brown', linewidth=2)
            axes[1, 2].set_title('Energy per Bit')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Energy/Bit (J/bit)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'network_performance.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_action_analysis(self, save: bool = True, show: bool = False):
        """Plot action distributions and analysis."""
        if not self.metrics['ee_actions'] or not self.metrics['ns_actions']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Action Analysis', fontsize=16)
        
        # Convert actions to numpy arrays
        ee_actions = np.array(self.metrics['ee_actions'])
        ns_actions = np.array(self.metrics['ns_actions'])
        
        # Energy efficiency actions
        axes[0, 0].hist(ee_actions[:, 0], bins=8, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('a_t (Normal PRB Duration)')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(ee_actions[:, 1], bins=8, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('b_t (Sleep Duration)')
        axes[0, 1].set_xlabel('Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].hist(ee_actions[:, 2], bins=8, alpha=0.7, color='red', edgecolor='black')
        axes[0, 2].set_title('c_t (Second Active Duration)')
        axes[0, 2].set_xlabel('Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Network slicing actions
        axes[1, 0].hist(ns_actions[:, 0], bins=20, alpha=0.7, color='cyan', edgecolor='black')
        axes[1, 0].set_title('Slice 1 (eMBB)')
        axes[1, 0].set_xlabel('Percentage')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].hist(ns_actions[:, 1], bins=20, alpha=0.7, color='magenta', edgecolor='black')
        axes[1, 1].set_title('Slice 2 (URLLC)')
        axes[1, 1].set_xlabel('Percentage')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        axes[1, 2].hist(ns_actions[:, 2], bins=20, alpha=0.7, color='yellow', edgecolor='black')
        axes[1, 2].set_title('Slice 3 (mMTC)')
        axes[1, 2].set_xlabel('Percentage')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'action_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multi_objective_analysis(self, save: bool = True, show: bool = False):
        """Plot multi-objective optimization analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Objective Analysis', fontsize=16)
        
        # Pareto efficiency
        if self.metrics['pareto_efficiency']:
            axes[0, 0].plot(self.metrics['epoch'], self.metrics['pareto_efficiency'], 
                           'blue', linewidth=2)
            axes[0, 0].set_title('Pareto Efficiency')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Efficiency Score')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Objective conflicts
        if self.metrics['objective_conflicts']:
            axes[0, 1].plot(self.metrics['epoch'], self.metrics['objective_conflicts'], 
                           'red', linewidth=2)
            axes[0, 1].set_title('Objective Conflicts')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Conflict Score')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Fairness index
        if self.metrics['fairness_index']:
            axes[1, 0].plot(self.metrics['epoch'], self.metrics['fairness_index'], 
                           'green', linewidth=2)
            axes[1, 0].set_title('Fairness Index')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Fairness Score')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Energy vs Performance trade-off
        if self.metrics['ep_ee_reward'] and self.metrics['ep_ns_reward']:
            axes[1, 1].scatter(self.metrics['ep_ee_reward'], self.metrics['ep_ns_reward'], 
                              alpha=0.6, c=self.metrics['epoch'], cmap='viridis')
            axes[1, 1].set_title('Energy vs Performance Trade-off')
            axes[1, 1].set_xlabel('Energy Efficiency Reward')
            axes[1, 1].set_ylabel('Network Slicing Reward')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1], label='Epoch')
        
        plt.tight_layout()
        if save:
            plt.savefig(os.path.join(self.save_dir, 'multi_objective_analysis.png'), 
                       dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_dashboard(self, save_html: bool = True):
        """Create interactive Plotly dashboard."""
        if not self.metrics['epoch']:
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Episode Returns', 'Training Losses', 'Network Performance',
                          'Energy Efficiency', 'Action Distributions', 'Multi-Objective'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Episode returns
        if self.metrics['ep_ret']:
            fig.add_trace(
                go.Scatter(x=self.metrics['epoch'], y=self.metrics['ep_ret'],
                          mode='lines', name='Episode Return', line=dict(color='blue')),
                row=1, col=1
            )
        
        # Training losses
        if self.metrics['pi_loss']:
            fig.add_trace(
                go.Scatter(x=self.metrics['epoch'], y=self.metrics['pi_loss'],
                          mode='lines', name='Policy Loss', line=dict(color='green')),
                row=1, col=2
            )
        if self.metrics['v_loss']:
            fig.add_trace(
                go.Scatter(x=self.metrics['epoch'], y=self.metrics['v_loss'],
                          mode='lines', name='Value Loss', line=dict(color='red')),
                row=1, col=2
            )
        
        # Network performance
        if self.metrics['throughput_embb']:
            fig.add_trace(
                go.Scatter(x=self.metrics['epoch'], y=self.metrics['throughput_embb'],
                          mode='lines', name='eMBB Throughput', line=dict(color='blue')),
                row=2, col=1
            )
        
        # Energy efficiency
        if self.metrics['energy_consumption']:
            fig.add_trace(
                go.Scatter(x=self.metrics['epoch'], y=self.metrics['energy_consumption'],
                          mode='lines', name='Energy Consumption', line=dict(color='purple')),
                row=2, col=2
            )
        
        # Action distributions
        if self.metrics['ee_actions']:
            ee_actions = np.array(self.metrics['ee_actions'])
            fig.add_trace(
                go.Histogram(x=ee_actions[:, 0], name='a_t', opacity=0.7),
                row=3, col=1
            )
        
        # Multi-objective
        if self.metrics['ep_ee_reward'] and self.metrics['ep_ns_reward']:
            fig.add_trace(
                go.Scatter(x=self.metrics['ep_ee_reward'], y=self.metrics['ep_ns_reward'],
                          mode='markers', name='EE vs NS', marker=dict(color=self.metrics['epoch'])),
                row=3, col=2
            )
        
        fig.update_layout(
            title_text="O-RAN PPO Training Dashboard",
            showlegend=True,
            height=1200
        )
        
        if save_html:
            fig.write_html(os.path.join(self.save_dir, 'training_dashboard.html'))
        
        return fig
    
    def save_metrics(self, filename: str = "training_metrics.json"):
        """Save all metrics to JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        
        # Convert numpy arrays to lists for JSON serialization
        data_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    data_to_save[key] = [v.tolist() for v in value]
                else:
                    data_to_save[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(data_to_save, f, indent=2)
        
        print(f"Metrics saved to {filepath}")
    
    def print_summary(self, epoch: int):
        """Print training summary."""
        if epoch % self.monitor_interval == 0:
            try:
                print(f"\n{'='*60}")
                print(f"Training Summary - Epoch {epoch}")
                print(f"{'='*60}")
                
                if self.metrics['ep_ret']:
                    print(f"Episode Return: {self.metrics['ep_ret'][-1]:.3f}")
                if self.metrics['pi_loss']:
                    print(f"Policy Loss: {self.metrics['pi_loss'][-1]:.6f}")
                if self.metrics['v_loss']:
                    print(f"Value Loss: {self.metrics['v_loss'][-1]:.6f}")
                if self.metrics['kl_div']:
                    print(f"KL Divergence: {self.metrics['kl_div'][-1]:.6f}")
                if self.metrics['energy_consumption']:
                    print(f"Energy Consumption: {self.metrics['energy_consumption'][-1]:.3f}")
                if self.metrics['qos_violations']:
                    print(f"QoS Violations: {self.metrics['qos_violations'][-1]}")
                
                print(f"{'='*60}")
            except Exception as e:
                print(f"Warning: Error printing summary for epoch {epoch}: {e}")
                # Continue with training even if summary printing fails
    
    def generate_all_plots(self, epoch: int, save: bool = True, show: bool = False):
        """Generate all plots for current epoch."""
        if epoch % 20 == 0 or epoch == 0:  # Generate plots every 20 epochs
            try:
                print(f"Generating plots for epoch {epoch}...")
                self.plot_training_progress(save=save, show=show)
                self.plot_network_performance(save=save, show=show)
                self.plot_action_analysis(save=save, show=show)
                self.plot_multi_objective_analysis(save=save, show=show)
                self.create_dashboard(save_html=True)
                print("All plots generated successfully!")
            except Exception as e:
                print(f"Warning: Error generating plots for epoch {epoch}: {e}")
                # Continue with training even if plot generation fails


# Utility functions for metric calculation
def calculate_network_metrics(env_state, actions):
    """Calculate network performance metrics from environment state."""
    # This should be implemented based on your environment
    # Placeholder implementation
    return {
        'throughput_embb': np.random.normal(500, 50),
        'throughput_urllc': np.random.normal(300, 30),
        'throughput_mmtc': np.random.normal(100, 20),
        'delay_embb': np.random.normal(15, 2),
        'delay_urllc': np.random.normal(5, 1),
        'delay_mmtc': np.random.normal(25, 5),
        'qos_violations': np.random.poisson(2),
        'resource_utilization': np.random.uniform(70, 90),
        'energy_consumption': np.random.normal(100, 10),
        'sleep_efficiency': np.random.uniform(0.6, 0.9),
        'energy_per_bit': np.random.normal(0.001, 0.0001)
    }

def calculate_action_metrics(ee_actions, ns_actions):
    """Calculate action-related metrics."""
    # Handle empty arrays
    if len(ee_actions) == 0 or len(ns_actions) == 0:
        return {
            'ee_actions': np.array([]),
            'ns_actions': np.array([]),
            'constraint_satisfaction': 0.0
        }
    
    ee_actions = np.array(ee_actions)
    ns_actions = np.array(ns_actions)
    
    # Check constraint satisfaction
    if ee_actions.size > 0 and ns_actions.size > 0:
        ee_sum = np.sum(ee_actions, axis=1)
        ns_sum = np.sum(ns_actions, axis=1)
        
        ee_constraint_satisfied = np.mean(np.abs(ee_sum - 7) < 0.1)
        ns_constraint_satisfied = np.mean(np.abs(ns_sum - 100) < 0.1)
        
        constraint_satisfaction = (ee_constraint_satisfied + ns_constraint_satisfied) / 2
    else:
        constraint_satisfaction = 0.0
    
    return {
        'ee_actions': ee_actions,
        'ns_actions': ns_actions,
        'constraint_satisfaction': constraint_satisfaction
    }

def calculate_multi_objective_metrics(ee_reward, ns_reward):
    """Calculate multi-objective optimization metrics."""
    # Handle edge cases where rewards might be zero
    if ee_reward == 0 and ns_reward == 0:
        return {
            'pareto_efficiency': 0.0,
            'objective_conflicts': 1.0,  # No conflicts when both are zero
            'fairness_index': 1.0  # Perfect fairness when both are zero
        }
    
    # Pareto efficiency (simplified)
    pareto_efficiency = (ee_reward + ns_reward) / 2
    
    # Objective conflicts (correlation-based)
    max_reward = max(abs(ee_reward), abs(ns_reward))
    if max_reward > 0:
        objective_conflicts = 1 - abs(ee_reward - ns_reward) / max_reward
    else:
        objective_conflicts = 1.0  # No conflicts when both are zero
    
    # Fairness index (Jain's fairness index)
    if ee_reward == 0 and ns_reward == 0:
        fairness_index = 1.0  # Perfect fairness when both are zero
    else:
        sum_squared = ee_reward**2 + ns_reward**2
        if sum_squared > 0:
            fairness_index = (ee_reward + ns_reward)**2 / (2 * sum_squared)
        else:
            fairness_index = 1.0
    
    return {
        'pareto_efficiency': pareto_efficiency,
        'objective_conflicts': objective_conflicts,
        'fairness_index': fairness_index
    }

# Example usage functions
def example_usage():
    """
    Example of how to use the visualization tools.
    """
    # Initialize visualizer
    visualizer = ORANVisualizer(save_dir="training_plots")
    
    # Simulate training data
    for epoch in range(100):
        metrics = {
            'ep_ret': np.random.normal(100, 20),
            'ep_len': np.random.normal(200, 30),
            'pi_loss': np.random.exponential(0.1),
            'v_loss': np.random.exponential(0.1),
            'kl_div': np.random.exponential(0.01),
            'entropy': np.random.normal(2.0, 0.5),
            'clip_frac': np.random.uniform(0, 0.3),
            'ep_ee_reward': np.random.normal(50, 10),
            'ep_ns_reward': np.random.normal(50, 10),
            'ep_total_reward': np.random.normal(100, 20),
            'ee_actions': np.random.randint(0, 8, (3,)),
            'ns_actions': np.random.uniform(20, 80, (3,)),
            'throughput_avg': np.random.normal(500, 100),
            'delay_avg': np.random.normal(10, 2),
            'energy_consumption': np.random.normal(100, 20)
        }
        
        visualizer.update_metrics(epoch, metrics, calculate_network_metrics(None, None), calculate_action_metrics(metrics['ee_actions'], metrics['ns_actions']))
    
    # Generate plots
    visualizer.generate_all_plots(0)


if __name__ == "__main__":
    example_usage() 