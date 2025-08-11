# EExApp - GNN-Based Reinforcement Learning for Radio Unit Energy Optimization in 5G O-RAN

## 📋 Overview

EExApp is an advanced Deep Reinforcement Learning (DRL) system that leverages Graph Neural Networks (GNN) to optimize energy efficiency in 5G Open Radio Access Network (O-RAN) infrastructure. The system implements a **Dual-Actor-Dual-Critic Proximal Policy Optimization (PPO)** algorithm enhanced with **Graph Attention Networks (GAT)** for joint optimization of transmission time scheduling and network slicing.

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
│  (17×N features)│    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                              │                 │
                                              ▼                 ▼
                                   ┌─────────────────┐    ┌─────────────────┐
                                   │   GAT Network   │    │  Constrained    │
                                   │   Aggregator    │    │  Action Space   │
                                   │                 │    │                 │
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
- **Input**: Variable-length UE states (17 KPM & MAC features per UE)
- **Architecture**: 2-layer transformer with 4 attention heads
- **Output**: Fixed-size encoded state (64 dimensions)
- **Purpose**: Process multi-UE network states efficiently

#### 2. **Dual Actor-Critic Architecture**
- **EE Actor**: Constrained categorical actor for energy efficiency
  - Output: 3 discrete actions `(a_t, b_t, c_t)` with sum constraint = 7
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


### Real O-RAN Deployment

Deploy trained model on real OAI 5G infrastructure:





## 🏛️ Project Structure

```
EExApp/
├── algorithms/                    # Core DRL implementation
│   ├── config.py                 # Centralized configuration
│   ├── env.py                    # O-RAN environment interface
│   ├── state_encoder.py          # Transformer state encoder
│   ├── gat.py                    # Graph Attention Network
│   ├── mcma_ppo.py              # Multi-Critic Multi-Actor PPO
│   └── ppo.py                   # Training and deployment logic
├── examples/                     # O-RAN integration examples
│   ├── xApp/                    # xApp implementations
│   └── ric/                     # RIC components
├── src/                         # Core O-RAN components
├── test/                        # Testing and validation
├── trandata/                    # Training data and metrics
├── requirements.txt             # Python dependencies
├── CMakeLists.txt              # Build configuration
└── README.md                   # This file
```

