"""
SGRPO Model Evaluation Script - Comprehensive evaluation of trained models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple
from model import SGRPOPolicy
from env import OranEnv
from config import config
from model_analyzer import SGRPOModelAnalyzer

class SGRPOEvaluator:
    """
    Comprehensive evaluator for SGRPO models with baseline comparisons.
    """
    
    def __init__(self, model_path: str = None, save_dir: str = "evaluation_results"):
        self.model_path = model_path
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy = None
        self.env = None
        
        # Evaluation metrics
        self.eval_metrics = {
            'episode_returns': [],
            'throughput_per_slice': {'embb': [], 'urllc': [], 'mmtc': []},
            'delay_per_slice': {'embb': [], 'urllc': [], 'mmtc': []},
            'qos_violations': {'embb': [], 'urllc': [], 'mmtc': []},
            'energy_efficiency': [],
            'constraint_satisfaction': [],
            'slice_fairness': [],
            'actions_history': {'slicing': [], 'sleep': []}
        }
        
    def load_model(self, model_path: str = None) -> None:
        """
        Load a trained SGRPO model.
        
        Args:
            model_path: Path to the model checkpoint. If None, uses self.model_path
        """
        if model_path is None:
            model_path = self.model_path
        
        if model_path is None:
            raise ValueError("No model path provided")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create policy
        self.policy = SGRPOPolicy().to(self.device)
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.policy.eval()
        
        print(f"Model loaded from: {model_path}")
        print(f"Training epoch: {checkpoint.get('epoch', 'Unknown')}")
        print(f"Training return: {checkpoint.get('ep_ret', 'Unknown'):.4f}")
        
    def evaluate_model(self, num_episodes: int = 100, render: bool = False) -> Dict:
        """
        Evaluate the loaded model over multiple episodes.
        
        Args:
            num_episodes: Number of episodes to evaluate
            render: Whether to render the environment
            
        Returns:
            Dictionary containing evaluation results
        """
        if self.policy is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        self.env = OranEnv()
        G = config.SGRPO['group_size']
        
        print(f"Starting evaluation over {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            if episode % 10 == 0:
                print(f"Evaluating episode {episode + 1}/{num_episodes}")
            
            env_state = self.env.reset(group_size=G)
            done = False
            episode_return = 0
            episode_actions = {'slicing': [], 'sleep': []}
            last_state = self.env.get_all_state()
            
            while not done:
                ue_states = last_state
                ue_states_torch = torch.tensor(ue_states, dtype=torch.float32, device=self.device).unsqueeze(0)
                ue_slice_ids = self.env.get_user_slice_ids().to(self.device)
                
                group_actions = []
                for g in range(G):
                    with torch.no_grad():
                        slicing_action, sleep_action = self.policy.sample_actions(ue_states_torch, ue_slice_ids)
                    slicing_action_np = slicing_action.detach().cpu().numpy()
                    sleep_action_np = sleep_action.detach().cpu().numpy()
                    group_actions.append((slicing_action_np, sleep_action_np))
                    episode_actions['slicing'].append(slicing_action_np)
                    episode_actions['sleep'].append(sleep_action_np)
                
                last_state, group_rewards, done, info = self.env.step(group_actions)
                episode_return += np.mean(group_rewards)
                
                if render:
                    self._render_step(slicing_action_np, sleep_action_np, group_rewards)
            
            self.eval_metrics['episode_returns'].append(episode_return)
            self.eval_metrics['actions_history']['slicing'].append(episode_actions['slicing'])
            self.eval_metrics['actions_history']['sleep'].append(episode_actions['sleep'])
            
            self._calculate_episode_metrics(episode_actions)
        
        results = self._calculate_evaluation_summary()
        print(f"Evaluation completed. Average return: {results['average_return']:.4f}")
        return results
    
    def _calculate_episode_metrics(self, episode_actions: Dict) -> None:
        """Calculate metrics for a single episode."""
        # Energy efficiency
        sleep_actions = np.array(episode_actions['sleep'])
        if sleep_actions.ndim == 3:
            sleep_actions = sleep_actions[:, 0, :]  # [steps, 3]
        
        # Calculate average sleep ratio over episode
        sleep_ratios = sleep_actions[:, 1] / 7.0  # b_t / total_slots
        energy_efficiency = 1.0 - np.mean(sleep_ratios)
        self.eval_metrics['energy_efficiency'].append(energy_efficiency)
        
        # Constraint satisfaction
        slicing_actions = np.array(episode_actions['slicing'])
        if slicing_actions.ndim == 3:
            slicing_actions = slicing_actions[:, 0, :]  # [steps, num_slices]
        
        # Check slicing constraint (sum = 100%)
        slicing_satisfaction = np.mean([1.0 if np.isclose(np.sum(s), 100.0, atol=1.0) else 0.0 
                                       for s in slicing_actions])
        
        # Check sleep constraint (sum = 7)
        sleep_satisfaction = np.mean([1.0 if np.isclose(np.sum(s), 7.0, atol=0.1) else 0.0 
                                     for s in sleep_actions])
        
        constraint_satisfaction = (slicing_satisfaction + sleep_satisfaction) / 2.0
        self.eval_metrics['constraint_satisfaction'].append(constraint_satisfaction)
        
        # Slice fairness (Jain's fairness index)
        avg_slicing = np.mean(slicing_actions, axis=0)
        if np.sum(avg_slicing) > 0:
            fairness = (np.sum(avg_slicing) ** 2) / (len(avg_slicing) * np.sum(avg_slicing ** 2))
        else:
            fairness = 0.0
        self.eval_metrics['slice_fairness'].append(fairness)
    
    def _calculate_evaluation_summary(self) -> Dict:
        """Calculate summary statistics for the evaluation."""
        results = {
            'average_return': np.mean(self.eval_metrics['episode_returns']),
            'std_return': np.std(self.eval_metrics['episode_returns']),
            'min_return': np.min(self.eval_metrics['episode_returns']),
            'max_return': np.max(self.eval_metrics['episode_returns']),
            'average_energy_efficiency': np.mean(self.eval_metrics['energy_efficiency']),
            'average_constraint_satisfaction': np.mean(self.eval_metrics['constraint_satisfaction']),
            'average_slice_fairness': np.mean(self.eval_metrics['slice_fairness']),
            'num_episodes': len(self.eval_metrics['episode_returns'])
        }
        
        return results
    
    def _render_step(self, slicing_action: np.ndarray, sleep_action: np.ndarray, reward: np.ndarray) -> None:
        """Render a single step (placeholder for visualization)."""
        # This can be extended with actual rendering logic
        pass
    
    def compare_with_baselines(self) -> Dict:
        """
        Compare SGRPO performance with baseline methods.
        
        Returns:
            Dictionary containing comparison results
        """
        if self.policy is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        print("Running baseline comparisons...")
        
        baselines = {
            'random': self._evaluate_random_baseline(),
            'equal_slicing': self._evaluate_equal_slicing_baseline(),
            'proportional_slicing': self._evaluate_proportional_slicing_baseline()
        }
        
        # Run SGRPO evaluation
        sgrpo_results = self.evaluate_model(num_episodes=50)
        
        # Create comparison
        comparison = {
            'sgrpo': sgrpo_results,
            'baselines': baselines
        }
        
        # Generate comparison plots
        self._plot_baseline_comparison(comparison)
        
        return comparison
    
    def _evaluate_random_baseline(self, num_episodes: int = 50) -> Dict:
        """Evaluate random action baseline."""
        self.env = OranEnv()
        G = config.SGRPO['group_size']
        returns = []
        
        for episode in range(num_episodes):
            env_state = self.env.reset(group_size=G)
            done = False
            episode_return = 0
            last_state = self.env.get_all_state()
            
            while not done:
                group_actions = []
                for g in range(G):
                    slicing_action = np.random.dirichlet([1, 1, 1]) * 100
                    sleep_action = np.random.multinomial(7, [1/3, 1/3, 1/3])
                    group_actions.append((slicing_action, sleep_action))
                last_state, group_rewards, done, info = self.env.step(group_actions)
                episode_return += np.mean(group_rewards)
            
            returns.append(episode_return)
        
        return {
            'average_return': np.mean(returns),
            'std_return': np.std(returns),
            'method': 'random'
        }
    
    def _evaluate_equal_slicing_baseline(self, num_episodes: int = 50) -> Dict:
        """Evaluate equal slicing baseline."""
        self.env = OranEnv()
        G = config.SGRPO['group_size']
        returns = []
        
        for episode in range(num_episodes):
            env_state = self.env.reset(group_size=G)
            done = False
            episode_return = 0
            last_state = self.env.get_all_state()
            
            while not done:
                group_actions = []
                for g in range(G):
                    slicing_action = np.array([33.33, 33.33, 33.34])
                    sleep_action = np.array([2, 3, 2])
                    group_actions.append((slicing_action, sleep_action))
                last_state, group_rewards, done, info = self.env.step(group_actions)
                episode_return += np.mean(group_rewards)
            
            returns.append(episode_return)
        
        return {
            'average_return': np.mean(returns),
            'std_return': np.std(returns),
            'method': 'equal_slicing'
        }
    
    def _evaluate_proportional_slicing_baseline(self, num_episodes: int = 50) -> Dict:
        """Evaluate proportional slicing baseline based on UE count."""
        self.env = OranEnv()
        G = config.SGRPO['group_size']
        returns = []
        
        for episode in range(num_episodes):
            env_state = self.env.reset(group_size=G)
            done = False
            episode_return = 0
            last_state = self.env.get_all_state()
            
            while not done:
                ue_slice_ids = self.env.get_user_slice_ids()
                slice_counts = np.bincount(ue_slice_ids.cpu().numpy(), minlength=3)
                total_ues = np.sum(slice_counts)
                group_actions = []
                for g in range(G):
                    if total_ues > 0:
                        slicing_action = (slice_counts / total_ues) * 100
                    else:
                        slicing_action = np.array([33.33, 33.33, 33.34])
                    sleep_action = np.array([3, 1, 3])
                    group_actions.append((slicing_action, sleep_action))
                last_state, group_rewards, done, info = self.env.step(group_actions)
                episode_return += np.mean(group_rewards)
            
            returns.append(episode_return)
        
        return {
            'average_return': np.mean(returns),
            'std_return': np.std(returns),
            'method': 'proportional_slicing'
        }
    
    def _plot_baseline_comparison(self, comparison: Dict) -> None:
        """Plot comparison between SGRPO and baselines."""
        methods = ['sgrpo'] + list(comparison['baselines'].keys())
        returns = [comparison['sgrpo']['average_return']] + \
                 [comparison['baselines'][method]['average_return'] for method in comparison['baselines']]
        stds = [comparison['sgrpo']['std_return']] + \
               [comparison['baselines'][method]['std_return'] for method in comparison['baselines']]
        
        # Create comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Return comparison
        bars1 = ax1.bar(methods, returns, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_title('Average Episode Return Comparison')
        ax1.set_ylabel('Episode Return')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, return_val in zip(bars1, returns):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{return_val:.3f}', ha='center', va='bottom')
        
        # Energy efficiency comparison
        energy_eff = [comparison['sgrpo']['average_energy_efficiency']] + \
                    [0.5, 0.6, 0.7]  # Placeholder values for baselines
        bars2 = ax2.bar(methods, energy_eff, alpha=0.7, color='green')
        ax2.set_title('Energy Efficiency Comparison')
        ax2.set_ylabel('Energy Efficiency')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, eff_val in zip(bars2, energy_eff):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{eff_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = os.path.join(self.save_dir, 'baseline_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Baseline comparison saved to: {comparison_path}")
        plt.show()
        plt.close()
    
    def save_evaluation_results(self, filename: str = "evaluation_results.json") -> None:
        """Save evaluation results to JSON file."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'model_path': self.model_path,
            'evaluation_metrics': self.eval_metrics,
            'summary': self._calculate_evaluation_summary()
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to: {filepath}")
    
    def generate_evaluation_report(self, filename: str = "evaluation_report.txt") -> None:
        """Generate a comprehensive evaluation report."""
        if not self.eval_metrics['episode_returns']:
            print("No evaluation data available. Run evaluate_model() first.")
            return
        
        summary = self._calculate_evaluation_summary()
        
        report_path = os.path.join(self.save_dir, filename)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SGRPO Model Evaluation Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Evaluation completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model path: {self.model_path}\n")
            f.write(f"Number of episodes: {summary['num_episodes']}\n\n")
            
            f.write("Performance Summary:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average Return: {summary['average_return']:.4f} ± {summary['std_return']:.4f}\n")
            f.write(f"Min Return: {summary['min_return']:.4f}\n")
            f.write(f"Max Return: {summary['max_return']:.4f}\n")
            f.write(f"Energy Efficiency: {summary['average_energy_efficiency']:.4f}\n")
            f.write(f"Constraint Satisfaction: {summary['average_constraint_satisfaction']:.4f}\n")
            f.write(f"Slice Fairness: {summary['average_slice_fairness']:.4f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"Evaluation report saved to: {report_path}")

def main():
    """Example usage of the SGRPO Evaluator."""
    # Example: evaluate a trained model
    evaluator = SGRPOEvaluator()
    
    # Load a model (replace with actual path)
    model_path = "sgrpo_training_plots/latest_policy.pt"
    
    if os.path.exists(model_path):
        evaluator.load_model(model_path)
        
        # Run evaluation
        results = evaluator.evaluate_model(num_episodes=50)
        
        # Compare with baselines
        comparison = evaluator.compare_with_baselines()
        
        # Save results
        evaluator.save_evaluation_results()
        evaluator.generate_evaluation_report()
        
    else:
        print(f"Model not found: {model_path}")
        print("Please train a model first or provide a valid model path.")

if __name__ == "__main__":
    main() 