# SGRPO Visualization: Standalone implementation
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy scalars to Python scalars
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    else:
        return obj

class SGRPOVisualizer:
    """
    Comprehensive visualizer for SGRPO training with joint slicing and sleep scheduling.
    """
    
    def __init__(self, save_dir: str = "sgrpo_training_plots", config=None):
        self.save_dir = save_dir
        self.config = config
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize metric tracking
        self.metrics = {
            # Training metrics
            'epoch': [], 'ep_ret': [], 'ep_len': [], 'pi_loss': [],
            'kl_div': [], 'entropy': [], 'clip_frac': [],
            'group_adv_mean': [], 'group_adv_std': [],
            
            # Network performance
            'throughput_embb': [], 'throughput_urllc': [], 'throughput_mmtc': [],
            'delay_embb': [], 'delay_urllc': [], 'delay_mmtc': [],
            'qos_violation_embb': [], 'qos_violation_urllc': [], 'qos_violation_mmtc': [],
            
            # Actions
            'slicing_actions': [], 'sleep_actions': [], 'constraint_satisfaction': [],
            
            # Energy efficiency
            'energy_efficiency': [], 'sleep_ratio': [], 'active_ratio': [],
            
            # Slice-specific metrics
            'slice_utilization': [], 'slice_fairness': [], 'slice_qos_satisfaction': [],
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
            for key in ['ep_ret', 'ep_len', 'pi_loss', 'kl_div', 
                       'entropy', 'clip_frac', 'group_adv_mean', 'group_adv_std']:
                if key in training_metrics and training_metrics[key] is not None:
                    self.metrics[key].append(training_metrics[key])
            
            # Network performance
            for key in ['throughput_embb', 'throughput_urllc', 'throughput_mmtc',
                       'delay_embb', 'delay_urllc', 'delay_mmtc',
                       'qos_violation_embb', 'qos_violation_urllc', 'qos_violation_mmtc']:
                if key in network_metrics and network_metrics[key] is not None:
                    self.metrics[key].append(network_metrics[key])
            
            # Actions (only store if provided)
            if action_metrics and 'slicing_actions' in action_metrics and action_metrics['slicing_actions'] is not None:
                self.metrics['slicing_actions'].append(action_metrics['slicing_actions'].tolist() if isinstance(action_metrics['slicing_actions'], np.ndarray) else action_metrics['slicing_actions'])
            if action_metrics and 'sleep_actions' in action_metrics and action_metrics['sleep_actions'] is not None:
                self.metrics['sleep_actions'].append(action_metrics['sleep_actions'].tolist() if isinstance(action_metrics['sleep_actions'], np.ndarray) else action_metrics['sleep_actions'])
                
            # Calculate additional metrics
            self._calculate_derived_metrics(epoch, training_metrics, network_metrics, action_metrics)
                
        except Exception as e:
            print(f"Warning: Error updating metrics for epoch {epoch}: {e}")

    def _calculate_derived_metrics(self, epoch, training_metrics, network_metrics, action_metrics):
        """Calculate derived metrics like energy efficiency, constraint satisfaction, etc."""
        try:
            # Energy efficiency (based on sleep actions)
            if 'sleep_actions' in action_metrics and action_metrics['sleep_actions'] is not None:
                sleep_actions = np.array(action_metrics['sleep_actions'])
                if sleep_actions.size > 0:  # Check if array is not empty
                    if sleep_actions.ndim == 2:
                        sleep_actions = sleep_actions[-1]  # Use last action
                    sleep_ratio = sleep_actions[1] / 7.0  # b_t / total_slots
                    active_ratio = (sleep_actions[0] + sleep_actions[2]) / 7.0  # (a_t + c_t) / total_slots
                    energy_efficiency = sleep_ratio  # Higher is better
                    
                    self.metrics['energy_efficiency'].append(float(energy_efficiency))
                    self.metrics['sleep_ratio'].append(float(sleep_ratio))
                    self.metrics['active_ratio'].append(float(active_ratio))
            
            # Constraint satisfaction
            if 'slicing_actions' in action_metrics and action_metrics['slicing_actions'] is not None:
                slicing_actions = np.array(action_metrics['slicing_actions'])
                if slicing_actions.size > 0:  # Check if array is not empty
                    if slicing_actions.ndim == 2:
                        slicing_actions = slicing_actions[-1]  # Use last action
                    slicing_sum = np.sum(slicing_actions)
                    slicing_constraint = 1.0 if np.isclose(slicing_sum, 100.0, atol=1.0) else 0.0
                    
                    if 'sleep_actions' in action_metrics and action_metrics['sleep_actions'] is not None:
                        sleep_actions = np.array(action_metrics['sleep_actions'])
                        if sleep_actions.size > 0:  # Check if array is not empty
                            if sleep_actions.ndim == 2:
                                sleep_actions = sleep_actions[-1]
                            sleep_sum = np.sum(sleep_actions)
                            sleep_constraint = 1.0 if np.isclose(sleep_sum, 7.0, atol=0.1) else 0.0
                            
                            constraint_satisfaction = (slicing_constraint + sleep_constraint) / 2.0
                            self.metrics['constraint_satisfaction'].append(float(constraint_satisfaction))
            
            # Slice utilization and fairness
            if 'slicing_actions' in action_metrics and action_metrics['slicing_actions'] is not None:
                slicing_actions = np.array(action_metrics['slicing_actions'])
                if slicing_actions.size > 0:  # Check if array is not empty
                    if slicing_actions.ndim == 2:
                        slicing_actions = slicing_actions[-1]
                    
                    # Utilization (how much of the 100% is used)
                    utilization = np.sum(slicing_actions) / 100.0
                    self.metrics['slice_utilization'].append(float(utilization))
                    
                    # Fairness (Jain's fairness index)
                    if np.sum(slicing_actions) > 0:
                        fairness = (np.sum(slicing_actions) ** 2) / (len(slicing_actions) * np.sum(slicing_actions ** 2))
                        self.metrics['slice_fairness'].append(float(fairness))
                    else:
                        self.metrics['slice_fairness'].append(0.0)
                    
                    # QoS satisfaction
                    qos_satisfaction = 0.0
                    if self.config:
                        slice_names = ['embb', 'urllc', 'mmtc']
                        for s in range(self.config.ENV['num_slices']):
                            thp_key = f'throughput_{slice_names[s]}'
                            delay_key = f'delay_{slice_names[s]}'
                            if thp_key in network_metrics and delay_key in network_metrics:
                                thp = network_metrics[thp_key]
                                delay = network_metrics[delay_key]
                                req_thp = self.config.ENV['qos_targets'][s]['throughput']
                                req_delay = self.config.ENV['qos_targets'][s]['delay']
                                
                                thp_sat = 1.0 if thp >= req_thp else thp / req_thp
                                delay_sat = 1.0 if delay <= req_delay else req_delay / delay
                                qos_satisfaction += (thp_sat + delay_sat) / 2.0
                        
                        qos_satisfaction /= self.config.ENV['num_slices']
                        self.metrics['slice_qos_satisfaction'].append(float(qos_satisfaction))
                    
        except Exception as e:
            print(f"Warning: Error calculating derived metrics for epoch {epoch}: {e}")

    def plot_comprehensive_training_analysis(self, save=True, show=False):
        """Create comprehensive training analysis plots."""
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('SGRPO Comprehensive Training Analysis', fontsize=16)
        
        # Row 1: Core Training Metrics
        if self.metrics['ep_ret']:
            # Use step indices for step-level metrics
            step_indices = list(range(len(self.metrics['ep_ret'])))
            axes[0, 0].plot(step_indices, self.metrics['ep_ret'], 'b-', linewidth=2, label='Episode Return')
            axes[0, 0].set_title('Episode Returns')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Return')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
        
        if self.metrics['pi_loss']:
            step_indices = list(range(len(self.metrics['pi_loss'])))
            axes[0, 1].plot(step_indices, self.metrics['pi_loss'], 'r-', linewidth=2, label='Policy Loss')
            axes[0, 1].set_title('Policy Loss')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
        
        if self.metrics['kl_div']:
            step_indices = list(range(len(self.metrics['kl_div'])))
            axes[0, 2].plot(step_indices, self.metrics['kl_div'], 'g-', linewidth=2, label='KL Divergence')
            axes[0, 2].axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Target KL')
            axes[0, 2].set_title('KL Divergence')
            axes[0, 2].set_xlabel('Step')
            axes[0, 2].set_ylabel('KL Divergence')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].legend()
        
        # Row 2: Network Performance (use epoch-level metrics if available)
        slice_names = ['embb', 'urllc', 'mmtc']
        colors = ['blue', 'red', 'green']
        
        # Throughput comparison
        for s, (name, color) in enumerate(zip(slice_names, colors)):
            thp_key = f'throughput_{name}'
            if self.metrics[thp_key] and len(self.metrics[thp_key]) == len(self.metrics['epoch']):
                axes[1, 0].plot(self.metrics['epoch'], self.metrics[thp_key], 
                               color=color, linewidth=2, label=f'{name.upper()}')
                if self.config:
                    req = self.config.ENV['qos_targets'][s]['throughput']
                    axes[1, 0].axhline(y=req, color=color, linestyle='--', alpha=0.7)
        axes[1, 0].set_title('Per-Slice Throughput')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Throughput (kbps)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Delay comparison
        for s, (name, color) in enumerate(zip(slice_names, colors)):
            delay_key = f'delay_{name}'
            if self.metrics[delay_key] and len(self.metrics[delay_key]) == len(self.metrics['epoch']):
                axes[1, 1].plot(self.metrics['epoch'], self.metrics[delay_key], 
                               color=color, linewidth=2, label=f'{name.upper()}')
                if self.config:
                    req = self.config.ENV['qos_targets'][s]['delay']
                    axes[1, 1].axhline(y=req, color=color, linestyle='--', alpha=0.7)
        axes[1, 1].set_title('Per-Slice Delay')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Delay (ms)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        
        # QoS violations
        for s, (name, color) in enumerate(zip(slice_names, colors)):
            qv_key = f'qos_violation_{name}'
            if self.metrics[qv_key] and len(self.metrics[qv_key]) == len(self.metrics['epoch']):
                axes[1, 2].plot(self.metrics['epoch'], self.metrics[qv_key], 
                               color=color, linewidth=2, label=f'{name.upper()}')
        axes[1, 2].set_title('QoS Violations')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Violation Rate')
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].legend()
        
        # Row 3: Energy Efficiency and Actions (use epoch-level metrics)
        if self.metrics['energy_efficiency'] and len(self.metrics['energy_efficiency']) == len(self.metrics['epoch']):
            axes[2, 0].plot(self.metrics['epoch'], self.metrics['energy_efficiency'], 
                           'purple', linewidth=2, label='Sleep Ratio')
            axes[2, 0].set_title('Sleep Ratio')
            axes[2, 0].set_xlabel('Epoch')
            axes[2, 0].set_ylabel('Sleep Ratio')
            axes[2, 0].grid(True, alpha=0.3)
            axes[2, 0].legend()
        
        if self.metrics['constraint_satisfaction'] and len(self.metrics['constraint_satisfaction']) == len(self.metrics['epoch']):
            axes[2, 1].plot(self.metrics['epoch'], self.metrics['constraint_satisfaction'], 
                           'orange', linewidth=2, label='Constraint Satisfaction')
            axes[2, 1].set_title('Constraint Satisfaction')
            axes[2, 1].set_xlabel('Epoch')
            axes[2, 1].set_ylabel('Satisfaction Rate')
            axes[2, 1].grid(True, alpha=0.3)
            axes[2, 1].legend()
        
        if self.metrics['slice_fairness'] and len(self.metrics['slice_fairness']) == len(self.metrics['epoch']):
            axes[2, 2].plot(self.metrics['epoch'], self.metrics['slice_fairness'], 
                           'brown', linewidth=2, label='Slice Fairness')
            axes[2, 2].set_title('Slice Fairness (Jain\'s Index)')
            axes[2, 2].set_xlabel('Epoch')
            axes[2, 2].set_ylabel('Fairness Index')
            axes[2, 2].grid(True, alpha=0.3)
            axes[2, 2].legend()
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.save_dir}/comprehensive_training_analysis.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_action_analysis(self, save=True, show=False):
        """Plot detailed action analysis."""
        if not self.metrics['slicing_actions'] or not self.metrics['sleep_actions']:
            print("Warning: No action data available for plotting")
            return
        
        # Check if we have enough data to plot
        if len(self.metrics['epoch']) < 1:
            print("Warning: Not enough epochs to plot action analysis")
            return
        
        # Check if action data dimensions match epoch dimensions
        num_epochs = len(self.metrics['epoch'])
        num_slicing_actions = len(self.metrics['slicing_actions'])
        num_sleep_actions = len(self.metrics['sleep_actions'])
        
        print(f"Debug: epochs={num_epochs}, slicing_actions={num_slicing_actions}, sleep_actions={num_sleep_actions}")
        
        # Use the minimum length to avoid dimension mismatch
        plot_length = min(num_epochs, num_slicing_actions, num_sleep_actions)
        if plot_length < 1:
            print("Warning: Not enough matching data points to plot")
            return
        
        # Create x-axis data (epochs) for plotting
        x_data = self.metrics['epoch'][:plot_length]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SGRPO Action Analysis', fontsize=16)
        
        # Convert actions to numpy arrays
        slicing_actions = np.array(self.metrics['slicing_actions'][:plot_length])
        sleep_actions = np.array(self.metrics['sleep_actions'][:plot_length])
        
        # Ensure we have the right shape
        if slicing_actions.ndim == 1:
            # If it's 1D, reshape to [epochs, features]
            slicing_actions = slicing_actions.reshape(1, -1)
        if sleep_actions.ndim == 1:
            # If it's 1D, reshape to [epochs, features]
            sleep_actions = sleep_actions.reshape(1, -1)
        
        # Slicing actions over time
        slice_names = ['eMBB', 'URLLC', 'mMTC']
        colors = ['blue', 'red', 'green']
        
        for i in range(min(3, slicing_actions.shape[1])):
            if slicing_actions.ndim == 3:
                data = slicing_actions[:, 0, i]  # [epochs, batch, slice]
            else:
                data = slicing_actions[:, i]  # [epochs, slice]
            axes[0, 0].plot(x_data, data, 
                           color=colors[i], linewidth=2, label=slice_names[i])
        axes[0, 0].set_title('Slicing Actions Over Time')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Resource Allocation (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Sleep actions over time
        sleep_names = ['a_t (Active 1)', 'b_t (Sleep)', 'c_t (Active 2)']
        for i in range(min(3, sleep_actions.shape[1])):
            if sleep_actions.ndim == 3:
                data = sleep_actions[:, 0, i]
            else:
                data = sleep_actions[:, i]
            axes[0, 1].plot(x_data, data, 
                           color=colors[i], linewidth=2, label=sleep_names[i])
        axes[0, 1].set_title('Sleep Actions Over Time')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Slot Count')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Action distributions (histograms)
        if slicing_actions.ndim == 3:
            slicing_flat = slicing_actions[:, 0, :].flatten()
            sleep_flat = sleep_actions[:, 0, :].flatten()
        else:
            slicing_flat = slicing_actions.flatten()
            sleep_flat = sleep_actions.flatten()
        
        axes[0, 2].hist(slicing_flat, bins=30, alpha=0.7, color='cyan', edgecolor='black')
        axes[0, 2].set_title('Slicing Action Distribution')
        axes[0, 2].set_xlabel('Resource Allocation (%)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].hist(sleep_flat, bins=20, alpha=0.7, color='magenta', edgecolor='black')
        axes[1, 0].set_title('Sleep Action Distribution')
        axes[1, 0].set_xlabel('Slot Count')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Constraint satisfaction over time
        if (self.metrics['constraint_satisfaction'] and 
            len(self.metrics['constraint_satisfaction']) >= plot_length and
            len(self.metrics['constraint_satisfaction']) == len(self.metrics['epoch'])):
            constraint_data = self.metrics['constraint_satisfaction'][:plot_length]
            axes[1, 1].plot(x_data, constraint_data, 
                           'orange', linewidth=2)
            axes[1, 1].set_title('Constraint Satisfaction')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Satisfaction Rate')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Energy efficiency vs performance trade-off
        if (self.metrics['energy_efficiency'] and self.metrics['ep_ret'] and 
            len(self.metrics['energy_efficiency']) >= plot_length and 
            len(self.metrics['energy_efficiency']) == len(self.metrics['epoch']) and
            len(self.metrics['ep_ret']) >= plot_length):
            energy_data = self.metrics['energy_efficiency'][:plot_length]
            return_data = self.metrics['ep_ret'][:plot_length]
            axes[1, 2].scatter(energy_data, return_data, 
                              alpha=0.6, c=x_data, cmap='viridis')
            axes[1, 2].set_title('Energy vs Performance Trade-off')
            axes[1, 2].set_xlabel('Energy Efficiency')
            axes[1, 2].set_ylabel('Episode Return')
            axes[1, 2].grid(True, alpha=0.3)
            plt.colorbar(axes[1, 2].collections[0], ax=axes[1, 2], label='Epoch')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.save_dir}/action_analysis.png', dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        plt.close()

    def plot_sgrpo_metrics(self, save=True, show=False):
        # Per-slice allocation over time
        if ('slicing_actions' in self.metrics and 
            len(self.metrics['slicing_actions']) > 0 and
            len(self.metrics['slicing_actions']) == len(self.metrics['epoch'])):
            slicing_actions = np.array(self.metrics['slicing_actions'])  # [epochs, num_slices]
            
            # Handle 1D case
            if slicing_actions.ndim == 1:
                slicing_actions = slicing_actions.reshape(1, -1)
            
            if slicing_actions.ndim == 3:
                slicing_actions = slicing_actions[:, 0, :]  # [epochs, num_slices]
            
            plt.figure(figsize=(10, 6))
            for i in range(slicing_actions.shape[1]):
                plt.plot(self.metrics['epoch'], slicing_actions[:, i], label=f'Slice {i}')
            plt.title('Per-slice Resource Allocation Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Resource (%)')
            plt.legend()
            if save:
                plt.savefig(f'{self.save_dir}/sgrpo_slice_allocation.png', dpi=200)
            if show:
                plt.show()
            plt.close()
        # Constraint satisfaction
        if ('constraint_satisfaction' in self.metrics and 
            len(self.metrics['constraint_satisfaction']) > 0 and
            len(self.metrics['constraint_satisfaction']) == len(self.metrics['epoch'])):
            plt.figure(figsize=(8, 4))
            plt.plot(self.metrics['epoch'], self.metrics['constraint_satisfaction'], label='Constraint Satisfaction')
            plt.title('Constraint Satisfaction Over Time')
            plt.xlabel('Epoch')
            plt.ylabel('Fraction')
            plt.legend()
            if save:
                plt.savefig(f'{self.save_dir}/sgrpo_constraint_satisfaction.png', dpi=200)
            if show:
                plt.show()
            plt.close()
        # Group advantage distribution
        if ('group_adv_mean' in self.metrics and 'group_adv_std' in self.metrics and 
            len(self.metrics['group_adv_mean']) > 0 and len(self.metrics['group_adv_std']) > 0 and
            len(self.metrics['group_adv_mean']) == len(self.metrics['epoch']) and
            len(self.metrics['group_adv_std']) == len(self.metrics['epoch'])):
            plt.figure(figsize=(8, 4))
            plt.plot(self.metrics['epoch'], self.metrics['group_adv_mean'], label='GroupAdv Mean')
            plt.plot(self.metrics['epoch'], self.metrics['group_adv_std'], label='GroupAdv Std')
            plt.title('Group Advantage Statistics')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            if save:
                plt.savefig(f'{self.save_dir}/sgrpo_group_adv_stats.png', dpi=200)
            if show:
                plt.show()
            plt.close()
        # Per-UE reward histogram (last epoch)
        if 'ep_ret' in self.metrics and len(self.metrics['ep_ret']) > 0:
            plt.figure(figsize=(8, 4))
            plt.hist(self.metrics['ep_ret'], bins=20, alpha=0.7)
            plt.title('Per-UE Reward Histogram (Last Epoch)')
            plt.xlabel('Reward')
            plt.ylabel('Count')
            if save:
                plt.savefig(f'{self.save_dir}/sgrpo_per_ue_reward_hist.png', dpi=200)
            if show:
                plt.show()
            plt.close()

    def plot_training_process(self, save=True, show=False):
        metrics = self.metrics
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        if metrics['ep_ret']:
            step_indices = list(range(len(metrics['ep_ret'])))
            plt.plot(step_indices, metrics['ep_ret'], label='Episode Return')
        plt.title('Episode Return')
        plt.xlabel('Step')
        plt.legend()
        plt.subplot(2, 2, 2)
        if metrics['pi_loss']:
            step_indices = list(range(len(metrics['pi_loss'])))
            plt.plot(step_indices, metrics['pi_loss'], label='Policy Loss')
        plt.title('Policy Loss')
        plt.xlabel('Step')
        plt.legend()
        plt.subplot(2, 2, 3)
        if metrics['kl_div']:
            step_indices = list(range(len(metrics['kl_div'])))
            plt.plot(step_indices, metrics['kl_div'], label='KL Divergence')
        plt.title('KL Divergence')
        plt.xlabel('Step')
        plt.legend()
        plt.subplot(2, 2, 4)
        if metrics['entropy']:
            step_indices = list(range(len(metrics['entropy'])))
            plt.plot(step_indices, metrics['entropy'], label='Policy Entropy')
        plt.title('Policy Entropy')
        plt.xlabel('Step')
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.save_dir}/training_process.png')
        if show:
            plt.show()
        plt.close()
        # Clipping fraction
        if metrics['clip_frac']:
            plt.figure()
            step_indices = list(range(len(metrics['clip_frac'])))
            plt.plot(step_indices, metrics['clip_frac'], label='Clipping Fraction')
            plt.title('Clipping Fraction')
            plt.xlabel('Step')
            plt.legend()
            if save:
                plt.savefig(f'{self.save_dir}/clip_fraction.png')
            if show:
                plt.show()
            plt.close()

    def plot_network_performance(self, config, save=True, show=False):
        metrics = self.metrics
        slice_names = ['embb', 'urllc', 'mmtc']
        # Throughput
        plt.figure(figsize=(10, 6))
        for s in range(config.ENV['num_slices']):
            thp = metrics.get(f'throughput_{slice_names[s]}', [])
            if thp and len(thp) == len(metrics['epoch']):
                req = config.ENV['qos_targets'][s]['throughput']
                plt.plot(metrics['epoch'], thp, label=f'{slice_names[s]} throughput')
                plt.axhline(req, linestyle='--', color=f'C{s}', label=f'{slice_names[s]} req')
        plt.title('Per-slice Throughput')
        plt.xlabel('Epoch')
        plt.legend()
        if save:
            plt.savefig(f'{self.save_dir}/per_slice_throughput.png')
        if show:
            plt.show()
        plt.close()
        # Delay
        plt.figure(figsize=(10, 6))
        for s in range(config.ENV['num_slices']):
            delay = metrics.get(f'delay_{slice_names[s]}', [])
            if delay and len(delay) == len(metrics['epoch']):
                req = config.ENV['qos_targets'][s]['delay']
                plt.plot(metrics['epoch'], delay, label=f'{slice_names[s]} delay')
                plt.axhline(req, linestyle='--', color=f'C{s}', label=f'{slice_names[s]} req')
        plt.title('Per-slice Delay')
        plt.xlabel('Epoch')
        plt.legend()
        if save:
            plt.savefig(f'{self.save_dir}/per_slice_delay.png')
        if show:
            plt.show()
        plt.close()
        # QoS violation
        plt.figure(figsize=(10, 6))
        for s in range(config.ENV['num_slices']):
            qv = metrics.get(f'qos_violation_{slice_names[s]}', [])
            if qv and len(qv) == len(metrics['epoch']):
                plt.plot(metrics['epoch'], qv, label=f'{slice_names[s]} QoS violation')
        plt.title('Per-slice QoS Violation')
        plt.xlabel('Epoch')
        plt.legend()
        if save:
            plt.savefig(f'{self.save_dir}/per_slice_qos_violation.png')
        if show:
            plt.show()
        plt.close()

    def save_detailed_metrics(self, filename: str = "sgrpo_detailed_metrics.json"):
        """Save detailed metrics to JSON file with metadata."""
        filepath = os.path.join(self.save_dir, filename)
        # Prepare data for JSON serialization
        data_to_save = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_epochs': len(set(self.metrics['epoch'])) if self.metrics['epoch'] else 0,  # Count unique epochs
                'config': self.config.__dict__ if self.config else None
            },
            'metrics': {}
        }
        # Convert numpy arrays to lists for JSON serialization
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    data_to_save['metrics'][key] = [v.tolist() for v in value]
                else:
                    data_to_save['metrics'][key] = value
        with open(filepath, 'w') as f:
            json.dump(to_serializable(data_to_save), f, indent=2)
        print(f"Detailed metrics saved to {filepath}")

    def create_training_summary(self, filename: str = "sgrpo_training_summary.txt"):
        """Create a comprehensive training summary report."""
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SGRPO Training Summary Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total epochs: {len(set(self.metrics['epoch'])) if self.metrics['epoch'] else 0}\n\n")  # Count unique epochs
            
            # Final metrics
            if self.metrics['ep_ret']:
                f.write(f"Final Episode Return: {self.metrics['ep_ret'][-1]:.4f}\n")
                f.write(f"Average Episode Return: {np.mean(self.metrics['ep_ret']):.4f}\n")
                f.write(f"Best Episode Return: {np.max(self.metrics['ep_ret']):.4f}\n\n")
            
            if self.metrics['pi_loss']:
                f.write(f"Final Policy Loss: {self.metrics['pi_loss'][-1]:.6f}\n")
                f.write(f"Average Policy Loss: {np.mean(self.metrics['pi_loss']):.6f}\n\n")
            
            if self.metrics['kl_div']:
                f.write(f"Final KL Divergence: {self.metrics['kl_div'][-1]:.6f}\n")
                f.write(f"Average KL Divergence: {np.mean(self.metrics['kl_div']):.6f}\n\n")
            
            # Network performance summary
            f.write("Network Performance Summary:\n")
            f.write("-" * 30 + "\n")
            slice_names = ['embb', 'urllc', 'mmtc']
            for s, name in enumerate(slice_names):
                thp_key = f'throughput_{name}'
                delay_key = f'delay_{name}'
                qv_key = f'qos_violation_{name}'
                
                if self.metrics[thp_key]:
                    f.write(f"{name.upper()} - Avg Throughput: {np.mean(self.metrics[thp_key]):.2f} kbps\n")
                if self.metrics[delay_key]:
                    f.write(f"{name.upper()} - Avg Delay: {np.mean(self.metrics[delay_key]):.2f} ms\n")
                if self.metrics[qv_key]:
                    f.write(f"{name.upper()} - Avg QoS Violation: {np.mean(self.metrics[qv_key]):.4f}\n")
                f.write("\n")
            
            # Energy efficiency
            if self.metrics['energy_efficiency']:
                f.write(f"Average Energy Efficiency: {np.mean(self.metrics['energy_efficiency']):.4f}\n")
                f.write(f"Final Energy Efficiency: {self.metrics['energy_efficiency'][-1]:.4f}\n\n")
            
            # Constraint satisfaction
            if self.metrics['constraint_satisfaction']:
                f.write(f"Average Constraint Satisfaction: {np.mean(self.metrics['constraint_satisfaction']):.4f}\n")
                f.write(f"Final Constraint Satisfaction: {self.metrics['constraint_satisfaction'][-1]:.4f}\n\n")
            
            f.write("=" * 60 + "\n")
            f.write("End of Report\n")
            f.write("=" * 60 + "\n")
        
        print(f"Training summary saved to {filepath}")

    def save_metrics(self, filename: str = "sgrpo_training_metrics.json"):
        """Save all metrics to JSON file."""
        filepath = os.path.join(self.save_dir, filename)
        data_to_save = {}
        for key, value in self.metrics.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    data_to_save[key] = [v.tolist() for v in value]
                else:
                    data_to_save[key] = value
        with open(filepath, 'w') as f:
            json.dump(to_serializable(data_to_save), f, indent=2)
        print(f"Metrics saved to {filepath}")

    def print_summary(self, epoch: int):
        """Print training summary every monitor_interval epochs."""
        if epoch % self.monitor_interval == 0:
            try:
                print(f"\n{'='*60}")
                print(f"SGRPO Training Summary - Epoch {epoch}")
                print(f"{'='*60}")
                if self.metrics['ep_ret']:
                    print(f"Episode Return: {self.metrics['ep_ret'][-1]:.3f}")
                if self.metrics['pi_loss']:
                    print(f"Policy Loss: {self.metrics['pi_loss'][-1]:.6f}")
                if self.metrics['kl_div']:
                    print(f"KL Divergence: {self.metrics['kl_div'][-1]:.6f}")
                if self.metrics['energy_efficiency']:
                    print(f"Energy Efficiency: {self.metrics['energy_efficiency'][-1]:.3f}")
                if self.metrics['constraint_satisfaction']:
                    print(f"Constraint Satisfaction: {self.metrics['constraint_satisfaction'][-1]:.3f}")
                if self.metrics['group_adv_mean']:
                    print(f"GroupAdv Mean: {self.metrics['group_adv_mean'][-1]:.4f}")
                if self.metrics['group_adv_std']:
                    print(f"GroupAdv Std: {self.metrics['group_adv_std'][-1]:.4f}")
                print(f"{'='*60}")
            except Exception as e:
                print(f"Warning: Error printing summary for epoch {epoch}: {e}")

    def generate_all_plots(self, epoch=None):
        """Generate all comprehensive plots and reports, and save metrics and summary."""
        print(f"Generating comprehensive plots and reports for epoch {epoch}...")
        self.plot_training_process(save=True, show=False)
        if self.config:
            self.plot_network_performance(self.config, save=True, show=False)
        self.plot_sgrpo_metrics(save=True, show=False)
        self.plot_action_analysis(save=True, show=False)
        self.plot_comprehensive_training_analysis(save=True, show=False)
        self.save_detailed_metrics()
        self.save_metrics()
        self.create_training_summary()
        print("All plots, metrics, and reports generated successfully!") 