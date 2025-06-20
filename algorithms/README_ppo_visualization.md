# PPO Training with Integrated Visualization

This document explains how to use the enhanced `ppo.py` file that now includes comprehensive visualization capabilities for O-RAN dual-actor PPO training.

## Overview

The enhanced `ppo.py` integrates the `visualization.py` module to provide real-time monitoring and visualization of:

- **Training Progress**: Episode returns, losses, KL divergence, entropy
- **Network Performance**: Throughput, delay, QoS violations, resource utilization
- **Energy Efficiency**: Energy consumption, sleep efficiency, energy per bit
- **Action Analysis**: Distribution of EE and NS actions, constraint satisfaction
- **Multi-Objective Metrics**: Pareto efficiency, objective conflicts, fairness

## Quick Start

### 1. Basic Usage

```python
from ppo import ppo_mcma
from env import OranEnv

# Set up environment
def env_fn():
    return OranEnv(
        num_slices=config.ENV['num_slices'],
        N_sf=config.ENV['N_sf'],
        user_num=config.ENV['user_num']
    )

# Run PPO with visualization
ppo_mcma(
    env_fn=env_fn,
    epochs=100,
    save_dir="my_training_results"
)
```

### 2. Using the Example Script

```bash
python example_ppo_with_viz.py
```

This will present you with options:
- Quick test (5 epochs)
- Full example (50 epochs)
- Check visualization files
- Run with command line arguments

### 3. Command Line Usage

```bash
python ppo.py --exp_name my_experiment --save_dir results
```

## Output Structure

After training, you'll find the following structure:

```
my_training_results/
├── models/
│   ├── mcma_ppo_epoch_20.pt
│   ├── mcma_ppo_epoch_40.pt
│   └── ...
├── plots/
│   ├── training_progress.png
│   ├── network_performance.png
│   ├── action_analysis.png
│   ├── multi_objective_analysis.png
│   ├── training_dashboard.html
│   └── ...
└── metrics_epoch_*.json
```

## Visualization Features

### 1. Training Progress Plots (`training_progress.png`)

- **Episode Returns**: Shows learning progress over time
- **Training Losses**: Policy and value function losses
- **KL Divergence**: Measures policy update magnitude
- **Reward Components**: Energy efficiency vs network slicing rewards
- **Policy Entropy**: Exploration vs exploitation balance
- **Clipping Fraction**: PPO clipping effectiveness

### 2. Network Performance Plots (`network_performance.png`)

- **Throughput per Slice**: eMBB, URLLC, mMTC performance
- **Delay per Slice**: Latency metrics for each slice
- **QoS Violations**: Number of service quality violations
- **Resource Utilization**: Overall resource usage efficiency
- **Energy Consumption**: Power usage over time
- **Energy per Bit**: Energy efficiency metric

### 3. Action Analysis Plots (`action_analysis.png`)

- **EE Action Distributions**: 
  - `a_t` (Normal PRB Duration)
  - `b_t` (Sleep Duration) 
  - `c_t` (Second Active Duration)
- **NS Action Distributions**:
  - Slice 1 (eMBB) percentage
  - Slice 2 (URLLC) percentage
  - Slice 3 (mMTC) percentage

### 4. Multi-Objective Analysis (`multi_objective_analysis.png`)

- **Pareto Efficiency**: Multi-objective optimization effectiveness
- **Objective Conflicts**: Trade-offs between energy and performance
- **Fairness Index**: Resource allocation fairness
- **Energy vs Performance Trade-off**: Scatter plot showing optimization frontier

### 5. Interactive Dashboard (`training_dashboard.html`)

A comprehensive Plotly-based interactive dashboard that combines all metrics with:
- Zoom and pan capabilities
- Hover information
- Interactive legends
- Export functionality

## Configuration

### Training Parameters

You can customize training parameters in `config.py` or pass them directly:

```python
ppo_mcma(
    env_fn=env_fn,
    epochs=200,                    # Number of training epochs
    steps_per_epoch=100,           # Steps per epoch
    save_freq=20,                  # Save model every N epochs
    save_dir="custom_results",     # Custom save directory
    target_kl=0.01,               # KL divergence target
    clip_ratio=0.2,               # PPO clipping ratio
    pi_lr=3e-4,                   # Policy learning rate
    vf_lr=1e-3                    # Value function learning rate
)
```

### Visualization Settings

The visualizer automatically:
- Generates plots every 20 epochs
- Saves metrics every `save_freq` epochs
- Creates an interactive dashboard at the end
- Prints training summaries every 10 epochs

## Monitoring Training

### Real-time Monitoring

During training, you'll see:
- Epoch progress and timing
- Average episode returns for both actors
- Loss values for policy and value functions
- Training summaries every 10 epochs

### Example Output

```
Epoch 10
Average EpRetEE: 45.234
Average EpRetNS: 52.876
Average EpLen: 156.7
Average EE Loss: 0.0234
Average NS Loss: 0.0187
TotalEnvInteracts: 1100
Time: 45.234

============================================================
Training Summary - Epoch 10
============================================================
Episode Return: 49.055
Policy Loss: 0.021050
Value Loss: 0.021050
KL Divergence: 0.008234
Energy Consumption: 98.456
QoS Violations: 3
============================================================
```

## Advanced Usage

### Custom Metrics

You can extend the visualization by modifying the metric calculation functions in `visualization.py`:

```python
def calculate_network_metrics(env_state, actions):
    """Custom network metric calculation"""
    # Implement your own metrics here
    return {
        'custom_metric_1': value1,
        'custom_metric_2': value2,
        # ... other metrics
    }
```

### Integration with Existing Code

If you have existing PPO training code, you can integrate visualization by adding:

```python
from visualization import ORANVisualizer

# Initialize visualizer
visualizer = ORANVisualizer(save_dir="plots")

# In your training loop
for epoch in range(epochs):
    # Your existing training code...
    
    # Add visualization
    visualizer.update_metrics(
        epoch=epoch,
        training_metrics=your_metrics,
        network_metrics=network_metrics,
        action_metrics=action_metrics
    )
    
    # Generate plots periodically
    if epoch % 20 == 0:
        visualizer.generate_all_plots(epoch)
```

## Troubleshooting

### Common Issues

1. **No plots generated**: Check that `matplotlib` and `plotly` are installed
2. **Memory issues**: Reduce `steps_per_epoch` or `epochs` for testing
3. **Import errors**: Ensure all dependencies are installed from `requirements_visualization.txt`

### Dependencies

Install required packages:

```bash
pip install -r requirements_visualization.txt
```

Required packages:
- matplotlib
- seaborn
- plotly
- pandas
- numpy
- torch

## Performance Considerations

- **Plot generation**: Plots are generated every 20 epochs to avoid slowing down training
- **Memory usage**: Large datasets may require reducing plot frequency
- **Disk space**: Plots and metrics are saved to disk; ensure sufficient space
- **GPU memory**: Visualization doesn't use GPU; training performance is unaffected

## Next Steps

1. Run a quick test with `example_ppo_with_viz.py`
2. Examine the generated plots and dashboard
3. Customize metrics for your specific use case
4. Integrate with your existing training pipeline

For more advanced visualization features, see the standalone `visualization.py` module documentation. 