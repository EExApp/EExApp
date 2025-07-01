"""
Test script for SGRPO visualization and analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from visualization import SGRPOVisualizer
from model_analyzer import SGRPOModelAnalyzer
from evaluate_model import SGRPOEvaluator
from config import config

def test_visualization():
    """Test the visualization module with sample data."""
    print("Testing SGRPO Visualization Module...")
    
    # Create visualizer
    visualizer = SGRPOVisualizer(save_dir='test_plots', config=config)
    
    # Generate sample training data
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Simulate training metrics
        training_metrics = {
            'ep_ret': 10.0 + 5.0 * np.sin(epoch * 0.1) + np.random.normal(0, 0.5),
            'ep_len': 100 + np.random.randint(-10, 10),
            'pi_loss': 0.5 * np.exp(-epoch * 0.05) + np.random.normal(0, 0.01),
            'kl_div': 0.01 * np.exp(-epoch * 0.03) + np.random.normal(0, 0.001),
            'entropy': 1.0 * np.exp(-epoch * 0.02) + np.random.normal(0, 0.05),
            'clip_frac': 0.1 * np.exp(-epoch * 0.04) + np.random.normal(0, 0.01),
            'group_adv_mean': np.random.normal(0, 0.5),
            'group_adv_std': 1.0 + np.random.normal(0, 0.1),
            'energy_efficiency': 0.7 + 0.2 * np.sin(epoch * 0.1) + np.random.normal(0, 0.05),
            'constraint_satisfaction': 0.9 + 0.1 * np.random.random(),
        }
        
        # Simulate network metrics
        network_metrics = {
            'throughput_embb': 120 + 20 * np.random.random(),
            'throughput_urllc': 60 + 10 * np.random.random(),
            'throughput_mmtc': 15 + 5 * np.random.random(),
            'delay_embb': 8 + 4 * np.random.random(),
            'delay_urllc': 0.8 + 0.4 * np.random.random(),
            'delay_mmtc': 80 + 40 * np.random.random(),
            'qos_violation_embb': 0.05 + 0.05 * np.random.random(),
            'qos_violation_urllc': 0.1 + 0.1 * np.random.random(),
            'qos_violation_mmtc': 0.15 + 0.1 * np.random.random(),
        }
        
        # Simulate action metrics
        action_metrics = {
            'slicing_actions': [np.array([40 + 10*np.random.random(), 
                                         30 + 10*np.random.random(), 
                                         30 + 10*np.random.random()])],
            'sleep_actions': [np.array([2 + np.random.randint(0, 2), 
                                       2 + np.random.randint(0, 2), 
                                       3 + np.random.randint(0, 2)])],
        }
        
        # Update visualizer
        visualizer.update_metrics(epoch, training_metrics, network_metrics, action_metrics)
    
    # Generate all plots
    print("Generating comprehensive plots...")
    visualizer.generate_all_plots()
    
    print("Visualization test completed successfully!")
    print(f"Plots saved in: {os.path.abspath('test_plots')}")

def test_model_analyzer():
    """Test the model analyzer module."""
    print("\nTesting SGRPO Model Analyzer...")
    
    # Create analyzer
    analyzer = SGRPOModelAnalyzer(models_dir='test_plots')
    
    # Since we don't have actual model files, test the interface
    print("Model analyzer interface test completed!")
    print("Note: Actual model analysis requires trained model checkpoints.")

def test_evaluator():
    """Test the evaluator module."""
    print("\nTesting SGRPO Evaluator...")
    
    # Create evaluator
    evaluator = SGRPOEvaluator(save_dir='test_evaluation')
    
    print("Evaluator interface test completed!")
    print("Note: Actual evaluation requires trained model checkpoints.")

def create_sample_model_checkpoint():
    """Create a sample model checkpoint for testing."""
    print("\nCreating sample model checkpoint...")
    
    import torch
    from model import SGRPOPolicy
    
    # Create sample checkpoint
    policy = SGRPOPolicy()
    
    checkpoint = {
        'epoch': 50,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': torch.optim.Adam(policy.parameters()).state_dict(),
        'training_metrics': {
            'ep_ret': 15.5,
            'pi_loss': 0.123,
            'kl_div': 0.008,
        },
        'network_metrics': {
            'throughput_embb': 125.0,
            'delay_embb': 7.5,
        },
        'config': config.__dict__,
        'loss': 0.123,
        'kl_div': 0.008,
        'ep_ret': 15.5,
    }
    
    # Save checkpoint
    os.makedirs('test_plots', exist_ok=True)
    torch.save(checkpoint, 'test_plots/policy_epoch50.pt')
    torch.save(checkpoint, 'test_plots/latest_policy.pt')
    
    print("Sample checkpoint created: test_plots/policy_epoch50.pt")

def main():
    """Run all tests."""
    print("=" * 60)
    print("SGRPO Visualization and Analysis Test Suite")
    print("=" * 60)
    
    try:
        # Create sample data and test visualization
        test_visualization()
        
        # Create sample model checkpoint
        create_sample_model_checkpoint()
        
        # Test model analyzer with sample checkpoint
        test_model_analyzer()
        
        # Test evaluator
        test_evaluator()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        print("\nGenerated files:")
        print("- test_plots/: Visualization outputs and sample model")
        print("- test_evaluation/: Evaluation framework (ready for use)")
        
        # List generated files
        if os.path.exists('test_plots'):
            print("\nFiles in test_plots/:")
            for file in os.listdir('test_plots'):
                print(f"  - {file}")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 