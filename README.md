# EExApp - GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red.svg)](https://pytorch.org/)

## 📋 Overview

EExApp is an advanced Deep Reinforcement Learning (DRL) system that leverages Graph Neural Networks (GNN) to optimize energy efficiency in 5G Open Radio Access Network (O-RAN) infrastructure. The system implements a **Multi-Critic Multi-Actor (MCMA) Proximal Policy Optimization (PPO)** algorithm enhanced with **Graph Attention Networks (GAT)** for joint optimization of transmission time scheduling and network slicing.

### 🎯 Key Objectives

- **Energy Efficiency**: Optimize Radio Unit (RU) power consumption through intelligent sleep scheduling
- **Network Slicing**: Dynamically allocate resources across different service types (eMBB, uRLLC, mMTC)
- **QoS Guarantee**: Maintain service quality while maximizing energy savings
- **Real-time Adaptation**: Near-real-time control for dynamic network conditions

## 🏗️ Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Multi-UE      │    │  Transformer    │    │   Dual Actors   │
│   State Input   │───▶│   State Encoder │───▶│  + Dual Critics │
│   (17×N features)│    │   (64 dim)      │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   GAT Network   │    │  Constrained    │
                       │   Aggregator    │    │  Action Space   │
                       │   (64 dim)      │    │                 │
                       └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │  Advantage      │    │  O-RAN Control  │
                       │  Calculation    │    │  Interface      │
                       └─────────────────┘    └─────────────────┘
```

### Network Structure

#### 1. **State Encoder (Transformer-based)**
- **Input**: Variable-length UE states (17 features per UE)
- **Architecture**: 2-layer transformer with 4 attention heads
- **Output**: Fixed-size encoded state (64 dimensions)
- **Purpose**: Process multi-UE network states efficiently

#### 2. **Dual Actor-Critic Architecture**
- **EE Actor**: Constrained categorical actor for energy efficiency
  - Output: 3 discrete actions `(a_t, b_t, c_t)` with sum constraint = 7
  - Constraint: `b_t < 5` for sleep period control
- **NS Actor**: Constrained Gaussian actor for network slicing
  - Output: 3 continuous actions `(slice1, slice2, slice3)` with sum = 100%
  - Purpose: Dynamic resource allocation across slices

#### 3. **GAT Aggregator**
- **Two-stage architecture**:
  - Critic-to-Critic GAT: Multi-head attention between critics
  - Critic-to-Actor GAT: Bipartite attention from critics to actors
- **Features**: 4 attention heads, learnable embeddings, dropout regularization

## 🚀 Features

### Advanced DRL Capabilities
- **Multi-Objective Learning**: Separate optimization for energy efficiency and QoS
- **Graph Attention Coordination**: Cross-influence between objectives via GAT
- **Constraint Satisfaction**: Specialized sampling for action space constraints
- **Transformer Encoding**: Handle variable number of UEs efficiently

### O-RAN Integration
- **E2 Interface Support**: Direct communication with gNB for metrics and control
- **Real-time Control**: 10ms subframe-aligned control periods
- **Multi-Slice Support**: eMBB, uRLLC, and mMTC slice optimization
- **FlexRIC Compatibility**: Integration with standard O-RAN components

### Performance Monitoring
- **Comprehensive Metrics**: Energy consumption, throughput, latency, fairness
- **Real-time Visualization**: Training progress and network performance plots
- **Constraint Tracking**: Monitor action space constraint satisfaction
- **Multi-objective Analysis**: Pareto efficiency and fairness indices

## 📦 Installation

### Prerequisites
- Python 3.8 or later
- PyTorch 2.4.0 or later
- Access to OAI 5G infrastructure (optional for simulation)

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/EExApp.git
   cd EExApp
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import torch; import gym; print('Installation successful!')"
   ```

## 🎮 Usage

### Training Mode

Train the MCMA-GAT-PPO model:

```bash
cd algorithms
python ppo.py
```

**Configuration**: Modify `config.py` for:
- Network parameters (number of UEs, slices)
- DRL hyperparameters (learning rates, batch sizes)
- GAT configuration (attention heads, hidden dimensions)
- Reward weights and QoS targets

### Simulation Mode

Run with simulated O-RAN environment:

```bash
python main_adapter.py --mode train --num_ues 5 --simulation
```

### Real O-RAN Deployment

Deploy trained model on real OAI 5G infrastructure:

```bash
python main_adapter.py \
    --checkpoint_path models/mcma_ppo_epoch_100.pth \
    --num_ues 5 \
    --e2_ip 192.168.1.100 \
    --e2_port 36422 \
    --mode deploy \
    --max_runtime 3600
```

### Evaluation Mode

Evaluate model performance:

```bash
python main_adapter.py \
    --checkpoint_path models/mcma_ppo_epoch_100.pth \
    --mode evaluate \
    --num_ues 5
```

## 🔧 Configuration

### Key Parameters

```python
# Environment Configuration
ENV = {
    'num_slices': 3,        # eMBB, uRLLC, mMTC
    'N_sf': 7,             # DL slots per frame
    'user_num': 1,         # Number of UEs
    'lambda_p': 0.7,       # Throughput penalty weight
    'lambda_d': 0.3,       # Delay penalty weight
}

# DRL Configuration
PPO = {
    'steps_per_epoch': 20,
    'epochs': 100,
    'gamma': 0.99,
    'clip_ratio': 0.2,
    'pi_lr': 3e-4,
    'vf_lr': 1e-3,
}

# GAT Configuration
GAT = {
    'hidden_dim': 64,
    'num_heads': 4,
    'dropout': 0.1,
}
```

## 📊 Performance Metrics

### Energy Efficiency
- **Sleep Efficiency**: Percentage of time RU spends in sleep mode
- **Energy per Bit**: Power consumption normalized by throughput
- **Power Savings**: Reduction in power consumption vs. baseline

### Network Performance
- **Throughput**: Per-slice and aggregate throughput
- **Latency**: End-to-end delay for delay-sensitive services
- **Block Error Rate**: Transmission reliability metrics
- **Fairness**: Resource allocation fairness across slices

### Multi-Objective Optimization
- **Pareto Efficiency**: Trade-off between energy and QoS objectives
- **Constraint Satisfaction**: Adherence to action space constraints
- **Convergence**: Training stability and convergence metrics

## 🔬 Algorithm Details

### MCMA-GAT-PPO Algorithm

1. **State Collection**: Gather multi-UE network states (MAC + KPM metrics)
2. **State Encoding**: Transform variable-length states to fixed-size representation
3. **Dual Evaluation**: Compute values using specialized EE and NS critics
4. **GAT Aggregation**: Aggregate critic values using graph attention
5. **Action Sampling**: Generate constrained actions for both actors
6. **Environment Interaction**: Apply actions and collect rewards
7. **Policy Update**: PPO-based policy and value function updates

### Reward Decomposition

```python
# Energy Efficiency Reward
reward_ee = b_t / N_sf  # Sleep period efficiency

# Network Slicing Reward (Penalty-based)
reward_ns = -λ_p * throughput_penalty - λ_d * delay_penalty
```

### Constraint Handling

- **EE Actions**: Temperature-scaled multinomial sampling with sum constraint
- **NS Actions**: Softmax normalization with percentage constraints
- **Gradient Clipping**: Prevents policy collapse during training

## 🏛️ Project Structure

```
EExApp/
├── algorithms/                    # Core DRL implementation
│   ├── config.py                 # Centralized configuration
│   ├── env.py                    # O-RAN environment interface
│   ├── state_encoder.py          # Transformer state encoder
│   ├── gat.py                    # Graph Attention Network
│   ├── mcma_ppo.py              # Multi-Critic Multi-Actor PPO
│   ├── ppo.py                   # Training and deployment logic
│   ├── kairos.py                # Alternative Kairos algorithm
│   └── visualization.py         # Training visualization tools
├── examples/                     # O-RAN integration examples
│   ├── xApp/                    # xApp implementations
│   └── ric/                     # RIC components
├── src/                         # Core O-RAN components
├── test/                        # Testing and validation
├── trandata/                    # Training data and metrics
├── training_plots/              # Generated visualizations
├── requirements.txt             # Python dependencies
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

## 🧪 Testing and Validation

### Unit Tests
```bash
cd test
python -m pytest test_*.py
```

### Integration Tests
```bash
# Test with simulated environment
python algorithms/env.py --test

# Test GAT aggregation
python algorithms/gat.py --test
```

### Performance Validation
```bash
# Run performance benchmarks
python algorithms/visualization.py --benchmark
```

## 📈 Results and Visualization

The system provides comprehensive visualization capabilities:

- **Training Progress**: Loss curves, reward evolution, convergence metrics
- **Network Performance**: Throughput, latency, and QoS metrics over time
- **Energy Analysis**: Power consumption patterns and efficiency gains
- **Action Analysis**: Actor behavior and constraint satisfaction
- **GAT Attention**: Attention weight visualization for interpretability

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run code formatting
black algorithms/

# Run linting
flake8 algorithms/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAirInterface Software Alliance** for OAI 5G implementation
- **O-RAN Alliance** for E2 interface specifications and architectural guidance
- **FlexRIC** for near-real-time RIC framework
- **PyTorch** and **Gym** communities for DRL tooling

## 📚 References

1. **O-RAN Architecture**: [O-RAN Alliance Specifications](https://www.o-ran.org/specifications)
2. **PPO Algorithm**: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
3. **Graph Attention Networks**: Veličković et al., "Graph Attention Networks" (2018)
4. **Multi-Critic Methods**: Haarnoja et al., "Soft Actor-Critic" (2018)

## 📞 Support

For questions, issues, or contributions:
- **Issues**: [GitHub Issues](https://github.com/yourusername/EExApp/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/EExApp/discussions)
- **Email**: your.email@institution.edu

---

**EExApp** - Empowering Energy-Efficient 5G O-RAN through Graph Neural Networks and Reinforcement Learning 🚀 
