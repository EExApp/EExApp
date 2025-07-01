# Train the model with enhanced visualization
python sgrpo.py

# Test visualization tools
python test_visualization.py

# Analyze saved models
python model_analyzer.py

# Evaluate trained model
python evaluate_model.py




# SGRPO (Spatial Group Relative Policy Optimization) for O-RAN

This directory contains a comprehensive implementation of Spatial Group Relative Policy Optimization (SGRPO) for energy-efficient O-RAN resource management, including joint slicing and sleep scheduling control.

## Overview

SGRPO is a variant of PPO that uses spatial group relative advantages instead of a separate value network. This implementation is specifically designed for O-RAN environments with:

- **Joint Control**: Simultaneous resource slicing and sleep scheduling
- **Transformer-based State Encoding**: Dynamic UE count handling with slice-aware pooling
- **Energy Efficiency**: Optimized sleep scheduling for power savings
- **QoS-Aware**: Per-slice throughput and delay requirements
- **Comprehensive Visualization**: Extensive plotting and analysis tools

## File Structure

```
SGRPO/
├── sgrpo.py                 # Main training script
├── model.py                 # SGRPO policy network implementation
├── state_encoder.py         # Transformer-based state encoder
├── env.py                   # O-RAN environment wrapper
├── config.py                # Configuration parameters
├── utils.py                 # Utility functions (normalization, KL divergence, etc.)
├── visualization.py         # Comprehensive visualization tools
├── model_analyzer.py        # Model analysis and comparison utilities
├── evaluate_model.py        # Model evaluation and baseline comparison
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Key Features

### 1. Enhanced Visualization (`visualization.py`)

The visualization module provides comprehensive plotting capabilities:

#### Training Analysis Plots
- **Comprehensive Training Analysis**: 3x3 grid showing training metrics, network performance, and energy efficiency
- **Action Analysis**: Detailed analysis of slicing and sleep actions over time
- **Training Process**: Core training metrics (loss, KL divergence, entropy)
- **Network Performance**: Per-slice throughput, delay, and QoS violations

#### Derived Metrics
- **Energy Efficiency**: Based on sleep action ratios
- **Constraint Satisfaction**: How well actions satisfy resource constraints
- **Slice Fairness**: Jain's fairness index for resource allocation
- **QoS Satisfaction**: Per-slice QoS requirement fulfillment

#### Data Export
- **JSON Metrics**: Detailed metrics saved in structured JSON format
- **Training Summary**: Comprehensive text reports with statistics
- **Epoch-specific Metrics**: Individual epoch performance data

### 2. Model Analysis (`model_analyzer.py`)

Comprehensive model analysis and comparison tools:

#### Model Loading
- Load individual checkpoints or all models in directory
- Extract training metrics and configuration
- Model performance comparison across epochs

#### Analysis Features
- **Model Comparison**: Compare multiple checkpoints side-by-side
- **Best Model Identification**: Find optimal model based on performance
- **Performance Reports**: Generate detailed model analysis reports
- **Inference Loading**: Load models for deployment

### 3. Model Evaluation (`evaluate_model.py`)

Comprehensive evaluation framework with baseline comparisons:

#### Evaluation Metrics
- Episode returns and statistics
- Energy efficiency measurements
- Constraint satisfaction rates
- Slice fairness indices
- Action distribution analysis

#### Baseline Comparisons
- **Random Baseline**: Random slicing and sleep actions
- **Equal Slicing**: Equal resource allocation (33.33% each)
- **Proportional Slicing**: UE-count-based proportional allocation

#### Reporting
- JSON evaluation results
- Comprehensive evaluation reports
- Baseline comparison plots

## Usage

### 1. Training

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python sgrpo.py
```

The training script will:
- Save model checkpoints every `save_freq` epochs
- Generate comprehensive plots and reports
- Save detailed metrics in JSON format
- Create training summaries

### 2. Model Analysis

```bash
# Analyze all saved models
python model_analyzer.py

# Or use in code
from model_analyzer import SGRPOModelAnalyzer

analyzer = SGRPOModelAnalyzer()
models = analyzer.load_all_checkpoints()
analyzer.compare_models([10, 20, 30, 40, 50])  # Compare specific epochs
best_model = analyzer.analyze_best_model()
```

### 3. Model Evaluation

```bash
# Evaluate a trained model
python evaluate_model.py

# Or use in code
from evaluate_model import SGRPOEvaluator

evaluator = SGRPOEvaluator("sgrpo_training_plots/latest_policy.pt")
evaluator.load_model()
results = evaluator.evaluate_model(num_episodes=100)
comparison = evaluator.compare_with_baselines()
evaluator.save_evaluation_results()
```

## Output Files

### Training Outputs (`sgrpo_training_plots/`)

#### Model Checkpoints
- `policy_epoch{N}.pt`: Complete model checkpoints with metadata
- `latest_policy.pt`: Most recent model for easy access

#### Visualization Plots
- `comprehensive_training_analysis.png`: 3x3 grid of all metrics
- `action_analysis.png`: Detailed action analysis
- `training_process.png`: Core training metrics
- `per_slice_throughput.png`: Slice-specific throughput
- `per_slice_delay.png`: Slice-specific delay
- `per_slice_qos_violation.png`: QoS violation rates
- `sgrpo_slice_allocation.png`: Resource allocation over time
- `sgrpo_constraint_satisfaction.png`: Constraint satisfaction
- `sgrpo_group_adv_stats.png`: Group advantage statistics

#### Data Files
- `sgrpo_detailed_metrics.json`: Complete training metrics
- `sgrpo_training_summary.txt`: Human-readable training summary
- `epoch_{N}_metrics.json`: Individual epoch metrics

#### Analysis Files
- `model_comparison.png`: Model performance comparison
- `sgrpo_model_report.txt`: Model analysis report

### Evaluation Outputs (`evaluation_results/`)

- `evaluation_results.json`: Complete evaluation metrics
- `evaluation_report.txt`: Evaluation summary report
- `baseline_comparison.png`: SGRPO vs baseline comparison

## Configuration

Key configuration parameters in `config.py`:

```python
SGRPO = {
    'pi_lr': 3e-4,           # Policy learning rate
    'epsilon': 0.2,          # PPO clipping parameter
    'beta_kl': 0.01,         # KL penalty coefficient
    'target_kl': 0.01,       # Target KL divergence
    'steps_per_epoch': 2048, # Steps per training epoch
    'epochs': 100,           # Total training epochs
    'save_freq': 10,         # Model save frequency
}

ENV = {
    'num_slices': 3,         # Number of network slices
    'qos_targets': {         # Per-slice QoS requirements
        0: {'throughput': 100, 'delay': 10},  # eMBB
        1: {'throughput': 50, 'delay': 1},    # URLLC
        2: {'throughput': 10, 'delay': 100},  # mMTC
    }
}
```

## Key Algorithms

### 1. SGRPO Algorithm
- **Spatial Group Normalization**: Normalize advantages across UEs in each step
- **Joint Policy**: Single policy for both slicing and sleep actions
- **Transformer Encoder**: Dynamic state encoding for variable UE counts
- **Constraint Handling**: Soft constraints with penalty terms

### 2. Action Space
- **Slicing Actions**: Continuous allocation percentages (sum to 100%)
- **Sleep Actions**: Discrete slot allocation (a_t, b_t, c_t summing to 7)

### 3. Reward Function
- **Throughput Reward**: Per-slice throughput satisfaction
- **Delay Penalty**: QoS violation penalties
- **Energy Efficiency**: Sleep action optimization
- **Constraint Penalty**: Resource allocation constraints

## Performance Metrics

### Training Metrics
- Episode returns and convergence
- Policy loss and KL divergence
- Group advantage statistics
- Clipping fraction and entropy

### Network Performance
- Per-slice throughput and delay
- QoS violation rates
- Resource utilization
- Slice fairness indices

### Energy Efficiency
- Sleep ratio optimization
- Active slot utilization
- Power consumption modeling
- Energy-performance trade-offs

## Dependencies

```
torch>=1.9.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Errors**: Reduce batch size or use CPU
2. **NaN Losses**: Check learning rate and gradient clipping
3. **Poor Convergence**: Adjust hyperparameters or reward scaling
4. **Constraint Violations**: Increase constraint penalty coefficients

### Performance Tips

1. **GPU Acceleration**: Use CUDA for faster training
2. **Batch Processing**: Increase batch size if memory allows
3. **Hyperparameter Tuning**: Experiment with learning rates and penalties
4. **Regular Checkpointing**: Save models frequently for analysis

## Contributing

To extend the SGRPO implementation:

1. **Add New Metrics**: Extend the visualization module
2. **New Baselines**: Add comparison methods in evaluator
3. **Environment Modifications**: Update the O-RAN environment
4. **Algorithm Improvements**: Enhance the SGRPO algorithm

## Citation

If you use this implementation, please cite:

```bibtex
@article{sgrpo_oran_2025,
  title={Spatial Group Relative Policy Optimization for Energy-Efficient O-RAN Resource Management},
  author={},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This implementation is provided for research purposes. Please ensure compliance with your institution's policies and relevant licenses. 