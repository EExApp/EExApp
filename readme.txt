# GAT-Enabled Multi-Critic-Multi-Actor PPO for OAI 5G

This repository contains the implementation of a Graph Attention Network (GAT) enabled multi-critic-multi-actor Proximal Policy Optimization (PPO) algorithm for joint time scheduling and network slicing in OpenAirInterface (OAI) 5G infrastructure.

## Overview

The algorithm uses a multi-critic-multi-actor approach with GAT to optimize:
1. **Time Scheduling**: Controls active and sleep periods for energy efficiency
2. **Network Slicing**: Allocates resources across different slices (uRLLC, eMBB, mMTC)

The implementation is designed to work with real OAI 5G infrastructure, including:
- OAI Core Network 5G (CN5G)
- OAI gNodeB (gNB)
- Commercial Off-The-Shelf User Equipment (COTS UE)

## Repository Structure

```
.
├── GNNRL                            # Source code for DRL algorithm
│   ├── env.py                       # Environment interface
│   ├── oran_real_env.py             # Real-world environment adapter
│   ├── e2_interface.py              # E2 interface for OAI communication
│   ├── gat_network.py               # GAT network implementation
│   ├── mcma_ppo.py                  # Multi-critic-multi-actor PPO
│   └── state_encoder.py             # State encoder and actor-critic models
├── examples                         # xApp for RAN slicing and monitoring
│   ├── ric                          # RIC including E2 interface
│   └── xApp                         # Source code for xApps
│       ├── c                        # C-based xApps
│       └── python3                  # Python-based xApps
├── trandata                         # Data storage
│   ├── KPM_UE.txt                   # KPM metrics
│   ├── slice_ctrl.bin               # Slice control binary
│   ├── xapp_db_                     # xApp database
│   ├── kpm.py                       # KPM data visualization
│   ├── mac.py                       # MAC layer data visualization
│   └── rewards.csv                  # Reward tracking
├── eexapp.py                        # Energy Efficient xApp implementation
├── main_adapter.py                  # Main adapter for OAI 5G
├── deploy.py                        # Simplified deployment script
└── README.md                        # This file
```

## Prerequisites

- Python 3.8 or later
- PyTorch 1.8 or later
- NumPy, Matplotlib, and other dependencies specified in `requirements.txt`
- Access to OAI 5G infrastructure with:
  - OAI CN5G
  - OAI gNB
  - COTS UE
- E2 interface enabled in OAI gNB

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/gat-mcma-ppo-oai.git
   cd gat-mcma-ppo-oai
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure proper permissions for the deployment script:
   ```bash
   chmod +x deploy.py
   ```

## Usage

### Deployment on Real OAI 5G Infrastructure

To deploy a trained model on real OAI 5G infrastructure:

```bash
./deploy.py --checkpoint path/to/checkpoint --num_ues 5 --e2_ip 192.168.1.100 --e2_port 36422 --mode deploy --runtime 3600
```

Parameters:
- `--checkpoint`: Path to the trained model checkpoint
- `--num_ues`: Number of UEs to consider
- `--e2_ip`: IP address of the E2 interface
- `--e2_port`: Port of the E2 interface
- `--mode`: Operation mode (`deploy`, `evaluate`, or `train`)
- `--runtime`: Maximum runtime in seconds (for deploy mode)
- `--control_interval`: Control interval in seconds

### Training with Real-World Environment Wrapper

To train using the real-world environment wrapper:

```bash
./deploy.py --checkpoint path/to/checkpoint --num_ues 5 --e2_ip 192.168.1.100 --e2_port 36422 --mode train
```

### Evaluation with Real-World Environment Wrapper

To evaluate a trained model using the real-world environment wrapper:

```bash
./deploy.py --checkpoint path/to/checkpoint --num_ues 5 --e2_ip 192.168.1.100 --e2_port 36422 --mode evaluate
```

### Advanced Usage with Main Adapter

For more advanced configuration, you can use the main adapter directly:

```bash
python main_adapter.py --checkpoint_path path/to/checkpoint --num_ues 5 --e2_ip 192.168.1.100 --e2_port 36422 --mode deploy --max_runtime 3600 --control_interval 0.1 --log_dir logs --results_dir results
```

## Integration with OAI 5G

The implementation interfaces with OAI 5G through:

1. **E2 Interface**: Communicates with gNB for metrics collection and control
2. **File-Based Communication**: Uses `slice_ctrl.bin` for slice parameter communication
3. **Database Access**: Reads MAC metrics from SQLite database
4. **KPM File Reading**: Reads KPM metrics from text files

### Communication Flow

1. The algorithm retrieves state information from:
   - E2 interface (KPM and MAC metrics)
   - SQLite database (fallback for MAC metrics)
   - KPM_UE.txt file (fallback for KPM metrics)

2. The algorithm selects actions for:
   - Time scheduling (active and sleep periods)
   - Network slicing (resource allocation across slices)

3. Actions are applied through:
   - E2 interface control messages
   - Writing to slice_ctrl.bin file (fallback)

## Troubleshooting

### Common Issues

1. **E2 Interface Connection Failed**:
   - Check IP address and port configuration
   - Ensure E2 interface is enabled in gNB
   - Verify network connectivity between the machine running the code and the gNB

2. **Cannot Read KPM Metrics**:
   - Ensure KPM_UE.txt file is being written correctly
   - Check permissions on the file

3. **Cannot Write Slice Parameters**:
   - Ensure slice_ctrl.bin file is accessible and writable
   - Check that the format is correct (4 integers)

### Logs and Debugging

- All logs are stored in the specified log directory (default: `logs/`)
- For detailed debugging, check:
  - `deployment.log`: Main deployment log
  - `e2_interface.log`: E2 interface communication log
  - `oran_real_env.log`: Real environment wrapper log

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAirInterface Software Alliance for OAI 5G implementation
- O-RAN Alliance for E2 interface specifications
