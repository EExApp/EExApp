# EExApp - Energy-Efficient xApp for O-RAN

A Deep Reinforcement Learning (DRL) enabled xApp for achieving energy-efficient O-RAN through joint transmission time scheduling and network slicing optimization.

## Project Overview

This project implements a multi-critic multi-actor DRL approach to optimize both energy efficiency and QoS in O-RAN networks. The system uses a Graph Attention Network (GAT) to aggregate feedback from multiple critics specialized in different network aspects (uRLLC, eMBB, mMTC, and energy efficiency) to guide two actors responsible for:

1. **Transmission Time Scheduling (EE Actor)** - Optimizes sleep periods to conserve energy
2. **Network Slicing (NS Actor)** - Dynamically allocates radio resources across network slices

## Architecture

The project is organized as follows:

```
EExApp/
├── algorithms/              # DRL models & training logic
│   ├── state_encoder.py     # GRU-based state encoder
│   ├── gat_network.py       # GAT-based critic aggregator
│   ├── mcma_ppo.py          # PPO algorithm with multi-critic, multi-actor
│   └── env.py               # Environment for O-RAN slicing & scheduling
├── xApp/                    # Python xApp for O-RAN control logic
│   ├── eexapp.py            # Main xApp logic
│   └── config.json          # Configuration file
├── ric/                     # Existing RIC and E2 agent logic (C or Python)
├── emulator/                # Optional RAN emulator setup for testing
└── README.md                # Project description and usage guide
```

## Features

- **Multi-Critic Multi-Actor DRL**: Uses specialized critics for different QoS domains
- **Graph Attention Network**: Dynamically learns how to aggregate critic feedback
- **Joint Optimization**: Simultaneously optimizes time scheduling and network slicing
- **FlexRIC Integration**: Interfaces with standard O-RAN near-RT RIC components
- **Real-time Control**: Operates in near-real-time for adaptive resource allocation
- **QoS-aware**: Considers slice-specific QoS requirements (uRLLC, eMBB, mMTC)

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Gym 0.21.0+
- FlexRIC (for integration with real O-RAN components)
- SQLite3
- NumPy, threading, etc.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/EExApp.git
   cd EExApp
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install FlexRIC (optional, for real O-RAN integration):
   Follow the instructions at [FlexRIC documentation](https://gitlab.eurecom.fr/mosaic5g/flexric)

## Usage

### Simulation Mode

To run the xApp in simulation mode (without actual RIC connection):

```bash
cd xApp
python eexapp.py --sim
```

### Production Mode

To run with a real FlexRIC-based O-RAN system:

```bash
cd xApp
python eexapp.py --config custom_config.json
```

### Configuration

The `config.json` file contains all configurable parameters, including:

- Network parameters (users, slices)
- DRL hyperparameters (learning rates, batch sizes, etc.)
- QoS requirements for each slice type
- Reward function weights
- File paths for integration

## Algorithm

The core algorithm is a Multi-Critic Multi-Actor Graph Attention Network PPO (MCMA-GAT-PPO) that:

1. Collects network state information from MAC and KPM metrics
2. Encodes the state using a GRU-based encoder
3. Evaluates the state using four critics (uRLLC, eMBB, mMTC, Energy)
4. Aggregates critic values using a Graph Attention Network
5. Uses two actors to make decisions for time scheduling and network slicing
6. Applies these decisions to the O-RAN environment
7. Collects rewards and updates the policy

## Integration with O-RAN

The xApp integrates with O-RAN components through:

1. **FlexRIC API**: For communication with the near-RT RIC
2. **E2 Service Models**: Using KPM for metrics and RC for control
3. **Binary File Interface**: A shared file mechanism for synchronization with environment

## Performance Metrics

The system optimizes for:

- Energy efficiency through maximizing sleep periods
- Throughput across all UEs and slices
- Low latency for delay-sensitive applications
- Fairness in resource allocation
- Block error rate minimization

## Customization

To customize the xApp for different network configurations:

1. Modify the `config.json` file to adjust parameters
2. Update QoS requirements based on your specific use cases
3. Tune reward weights to prioritize different objectives

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenAirInterface for the OAI CN5G implementation
- FlexRIC for the near-real-time RIC framework
- The O-RAN Alliance for specifications and architectural guidance
