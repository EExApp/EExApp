"""
SGRPO Model Analyzer - Utility for loading and analyzing saved models
"""

import torch
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from model import SGRPOPolicy
from config import config
from visualization import SGRPOVisualizer

class SGRPOModelAnalyzer:
    """
    Utility class for analyzing saved SGRPO models and their performance.
    """
    
    def __init__(self, models_dir: str = "sgrpo_training_plots"):
        self.models_dir = models_dir
        self.models_data = {}
        
    def load_model_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load a model checkpoint and return the data.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            
        Returns:
            Dictionary containing model data
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model data
        model_data = {
            'epoch': checkpoint.get('epoch', 0),
            'model_state_dict': checkpoint.get('model_state_dict', {}),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict', {}),
            'training_metrics': checkpoint.get('training_metrics', {}),
            'network_metrics': checkpoint.get('network_metrics', {}),
            'config': checkpoint.get('config', {}),
            'loss': checkpoint.get('loss', 0.0),
            'kl_div': checkpoint.get('kl_div', 0.0),
            'ep_ret': checkpoint.get('ep_ret', 0.0),
            'checkpoint_path': checkpoint_path
        }
        
        return model_data
    
    def load_all_checkpoints(self) -> Dict[int, Dict]:
        """
        Load all checkpoint files in the models directory.
        
        Returns:
            Dictionary mapping epoch numbers to model data
        """
        checkpoints = {}
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.pt') and 'policy_epoch' in filename:
                try:
                    # Extract epoch number
                    epoch_str = filename.replace('policy_epoch', '').replace('.pt', '')
                    epoch = int(epoch_str)
                    
                    checkpoint_path = os.path.join(self.models_dir, filename)
                    model_data = self.load_model_checkpoint(checkpoint_path)
                    checkpoints[epoch] = model_data
                    
                except (ValueError, Exception) as e:
                    print(f"Warning: Could not load {filename}: {e}")
        
        # Sort by epoch
        self.models_data = dict(sorted(checkpoints.items()))
        return self.models_data
    
    def compare_models(self, epochs: Optional[List[int]] = None) -> None:
        """
        Compare multiple models and generate comparison plots.
        
        Args:
            epochs: List of epochs to compare. If None, compares all loaded models.
        """
        if not self.models_data:
            self.load_all_checkpoints()
        
        if not self.models_data:
            print("No models found to compare.")
            return
        
        if epochs is None:
            epochs = list(self.models_data.keys())
        
        # Filter models by requested epochs
        models_to_compare = {epoch: self.models_data[epoch] for epoch in epochs if epoch in self.models_data}
        
        if not models_to_compare:
            print("No models found for the specified epochs.")
            return
        
        # Create comparison plots
        self._plot_model_comparison(models_to_compare)
    
    def _plot_model_comparison(self, models: Dict[int, Dict]) -> None:
        """Create comparison plots for multiple models."""
        epochs = list(models.keys())
        
        # Extract metrics for comparison
        losses = [models[epoch]['loss'] for epoch in epochs]
        kl_divs = [models[epoch]['kl_div'] for epoch in epochs]
        ep_rets = [models[epoch]['ep_ret'] for epoch in epochs]
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SGRPO Model Comparison', fontsize=16)
        
        # Loss comparison
        axes[0, 0].plot(epochs, losses, 'ro-', linewidth=2, markersize=8)
        axes[0, 0].set_title('Policy Loss Comparison')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KL divergence comparison
        axes[0, 1].plot(epochs, kl_divs, 'go-', linewidth=2, markersize=8)
        axes[0, 1].set_title('KL Divergence Comparison')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('KL Divergence')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode return comparison
        axes[1, 0].plot(epochs, ep_rets, 'bo-', linewidth=2, markersize=8)
        axes[1, 0].set_title('Episode Return Comparison')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Episode Return')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Performance summary table
        axes[1, 1].axis('off')
        summary_text = "Model Performance Summary:\n\n"
        for epoch in epochs:
            model = models[epoch]
            summary_text += f"Epoch {epoch}:\n"
            summary_text += f"  Loss: {model['loss']:.6f}\n"
            summary_text += f"  KL: {model['kl_div']:.6f}\n"
            summary_text += f"  Return: {model['ep_ret']:.4f}\n\n"
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = os.path.join(self.models_dir, 'model_comparison.png')
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Model comparison saved to: {comparison_path}")
        plt.show()
        plt.close()
    
    def analyze_best_model(self) -> Optional[Dict]:
        """
        Find and analyze the best performing model based on episode return.
        
        Returns:
            Dictionary containing the best model data
        """
        if not self.models_data:
            self.load_all_checkpoints()
        
        if not self.models_data:
            print("No models found to analyze.")
            return None
        
        # Find best model by episode return
        best_epoch = max(self.models_data.keys(), key=lambda e: self.models_data[e]['ep_ret'])
        best_model = self.models_data[best_epoch]
        
        print(f"Best model found at epoch {best_epoch}:")
        print(f"  Episode Return: {best_model['ep_ret']:.4f}")
        print(f"  Policy Loss: {best_model['loss']:.6f}")
        print(f"  KL Divergence: {best_model['kl_div']:.6f}")
        print(f"  Checkpoint: {best_model['checkpoint_path']}")
        
        return best_model
    
    def load_model_for_inference(self, checkpoint_path: str) -> SGRPOPolicy:
        """
        Load a model checkpoint for inference.
        
        Args:
            checkpoint_path: Path to the .pt checkpoint file
            
        Returns:
            Loaded SGRPO policy model
        """
        model_data = self.load_model_checkpoint(checkpoint_path)
        
        # Create model instance
        policy = SGRPOPolicy()
        
        # Load state dict
        policy.load_state_dict(model_data['model_state_dict'])
        policy.eval()  # Set to evaluation mode
        
        print(f"Model loaded from epoch {model_data['epoch']}")
        print(f"Model performance: Return={model_data['ep_ret']:.4f}, Loss={model_data['loss']:.6f}")
        
        return policy
    
    def generate_model_report(self, output_path: str = "sgrpo_model_report.txt") -> None:
        """
        Generate a comprehensive report of all loaded models.
        
        Args:
            output_path: Path to save the report
        """
        if not self.models_data:
            self.load_all_checkpoints()
        
        if not self.models_data:
            print("No models found to report.")
            return
        
        report_path = os.path.join(self.models_dir, output_path)
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SGRPO Model Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Report generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total models analyzed: {len(self.models_data)}\n\n")
            
            # Model summary table
            f.write("Model Summary:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Epoch':<8} {'Return':<12} {'Loss':<12} {'KL Div':<12} {'File':<30}\n")
            f.write("-" * 80 + "\n")
            
            for epoch in sorted(self.models_data.keys()):
                model = self.models_data[epoch]
                filename = os.path.basename(model['checkpoint_path'])
                f.write(f"{epoch:<8} {model['ep_ret']:<12.4f} {model['loss']:<12.6f} "
                       f"{model['kl_div']:<12.6f} {filename:<30}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            
            # Best model analysis
            best_model = self.analyze_best_model()
            if best_model:
                f.write("\nBest Model Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Epoch: {best_model['epoch']}\n")
                f.write(f"Episode Return: {best_model['ep_ret']:.4f}\n")
                f.write(f"Policy Loss: {best_model['loss']:.6f}\n")
                f.write(f"KL Divergence: {best_model['kl_div']:.6f}\n")
                f.write(f"Checkpoint: {best_model['checkpoint_path']}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")
        
        print(f"Model report saved to: {report_path}")

def main():
    """Example usage of the SGRPO Model Analyzer."""
    analyzer = SGRPOModelAnalyzer()
    
    # Load all checkpoints
    print("Loading all model checkpoints...")
    models = analyzer.load_all_checkpoints()
    
    if models:
        print(f"Loaded {len(models)} model checkpoints")
        
        # Generate comprehensive report
        analyzer.generate_model_report()
        
        # Compare models (last 5 epochs)
        recent_epochs = sorted(models.keys())[-5:]
        if recent_epochs:
            print(f"Comparing models from epochs: {recent_epochs}")
            analyzer.compare_models(recent_epochs)
        
        # Analyze best model
        best_model = analyzer.analyze_best_model()
        
    else:
        print("No model checkpoints found.")

if __name__ == "__main__":
    main() 