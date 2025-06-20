#!/usr/bin/env python3
"""
Multi-Critic Multi-Actor PPO Implementation for 5G O-RAN Network Slicing

PPO training loop implementation for the Multi-Critic Multi-Actor architecture:
1. Observe network state
2. Encode state s_t using GRU  
3. Sample actions a_t = {a_t_1, a_t_2} using actors, collect transitions (s_t, a_t, V_t, s_(t+1), r_t, logp_t)
4. Aggregate critic values v_agg_t={v_agg_1_t, v_agg_2_t} from two critics to two actors using bipartite GAT
5. Compute advantage A_t of each actor based on v_agg_t, using GAE
6. Compute target return R_t of each critic based on advantage and value function
7. Optimize clipped objective function for each actor, and update each actor
8. Update each critic using Huber loss of target return and critic value

This implementation features:
- Two Critics: QoS-aware and Energy-aware value estimation
- Two Actors: EE (Energy Efficiency) discrete and NS (Network Slicing) continuous
- GAT Aggregator: Bipartite Graph Attention for critic-actor coordination before advantage computation
- GRU State Encoder: Processes multi-UE 5G network states

- Integrated Visualization: Comprehensive training progress and performance monitoring
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import time
from collections import deque
import logging
from typing import Dict, List, Tuple, Any, Optional
import os
import json

from mcma_ppo import MCMA_ActorCritic, discount_cumsum
from env import OranEnv
from config import config
from state_encoder import StateEncoder
from gat import BipartiteGAT
from visualization import ORANVisualizer, calculate_network_metrics, calculate_action_metrics, calculate_multi_objective_metrics

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class MCMABuffer:
    """
    Buffer for MCMA-PPO: stores all necessary data for two-actor, two-critic PPO with GAT aggregation.
    """
    def __init__(self, max_ues, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = []  # list of raw multi-UE states
        self.encoded_buf = np.zeros((size, config.STATE_ENCODER['hidden_dim']), dtype=np.float32)
        # EE actions are discrete (integers)
        self.ee_act_buf = np.zeros((size, config.ACTOR_CRITIC['ee_action_dim']), dtype=np.int32)
        # NS actions are continuous (floats)
        self.ns_act_buf = np.zeros((size, config.ACTOR_CRITIC['ns_action_dim']), dtype=np.float32)
        self.ee_logp_buf = np.zeros(size, dtype=np.float32)
        self.ns_logp_buf = np.zeros(size, dtype=np.float32)
        self.ee_val_buf = np.zeros(size, dtype=np.float32)
        self.ns_val_buf = np.zeros(size, dtype=np.float32)
        self.ee_agg_val_buf = np.zeros(size, dtype=np.float32)
        self.ns_agg_val_buf = np.zeros(size, dtype=np.float32)
        self.ee_rew_buf = np.zeros(size, dtype=np.float32)
        self.ns_rew_buf = np.zeros(size, dtype=np.float32)
        
        # Initialize advantage and return buffers
        self.ee_adv_buf = np.zeros(size, dtype=np.float32)
        self.ns_adv_buf = np.zeros(size, dtype=np.float32)
        self.ee_ret_buf = np.zeros(size, dtype=np.float32)
        self.ns_ret_buf = np.zeros(size, dtype=np.float32)
        
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.gamma, self.lam = gamma, lam

    def store(self, obs, encoded, ee_act, ns_act, ee_logp, ns_logp, ee_val, ns_val, ee_agg_val, ns_agg_val, ee_rew, ns_rew):
        assert self.ptr < self.max_size
        self.obs_buf.append(obs)
        self.encoded_buf[self.ptr] = encoded
        self.ee_act_buf[self.ptr] = ee_act
        self.ns_act_buf[self.ptr] = ns_act
        self.ee_logp_buf[self.ptr] = ee_logp
        self.ns_logp_buf[self.ptr] = ns_logp
        self.ee_val_buf[self.ptr] = ee_val
        self.ns_val_buf[self.ptr] = ns_val
        self.ee_agg_val_buf[self.ptr] = ee_agg_val
        self.ns_agg_val_buf[self.ptr] = ns_agg_val
        self.ee_rew_buf[self.ptr] = ee_rew
        self.ns_rew_buf[self.ptr] = ns_rew
        self.ptr += 1

    def finish_path(self, last_ee_val=0, last_ns_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        # GAE for EE
        rews = np.append(self.ee_rew_buf[path_slice], last_ee_val)
        vals = np.append(self.ee_agg_val_buf[path_slice], last_ee_val)
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        ee_adv = discount_cumsum(deltas, self.gamma * self.lam)
        ee_ret = discount_cumsum(rews, self.gamma)[:-1]
        # GAE for NS
        rews_ns = np.append(self.ns_rew_buf[path_slice], last_ns_val)
        vals_ns = np.append(self.ns_agg_val_buf[path_slice], last_ns_val)
        deltas_ns = rews_ns[:-1] + self.gamma * vals_ns[1:] - vals_ns[:-1]
        ns_adv = discount_cumsum(deltas_ns, self.gamma * self.lam)
        ns_ret = discount_cumsum(rews_ns, self.gamma)[:-1]
        # Store
        self.ee_adv_buf[path_slice] = ee_adv
        self.ns_adv_buf[path_slice] = ns_adv
        self.ee_ret_buf[path_slice] = ee_ret
        self.ns_ret_buf[path_slice] = ns_ret
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # Use available data even if buffer is not exactly full
        available_size = min(self.ptr, self.max_size)
        if available_size == 0:
            raise ValueError("Buffer is empty, cannot get data")
        
        self.ptr, self.path_start_idx = 0, 0
        # Normalize advantages
        ee_adv = self.ee_adv_buf[:available_size]
        ns_adv = self.ns_adv_buf[:available_size]
        ee_adv = (ee_adv - ee_adv.mean()) / (ee_adv.std() + 1e-8)
        ns_adv = (ns_adv - ns_adv.mean()) / (ns_adv.std() + 1e-8)
        data = dict(
            encoded=torch.as_tensor(self.encoded_buf[:available_size], dtype=torch.float32),
            ee_act=torch.as_tensor(self.ee_act_buf[:available_size], dtype=torch.long),
            ns_act=torch.as_tensor(self.ns_act_buf[:available_size], dtype=torch.float32),
            ee_logp=torch.as_tensor(self.ee_logp_buf[:available_size], dtype=torch.float32),
            ns_logp=torch.as_tensor(self.ns_logp_buf[:available_size], dtype=torch.float32),
            ee_adv=torch.as_tensor(ee_adv, dtype=torch.float32),
            ns_adv=torch.as_tensor(ns_adv, dtype=torch.float32),
            ee_ret=torch.as_tensor(self.ee_ret_buf[:available_size], dtype=torch.float32),
            ns_ret=torch.as_tensor(self.ns_ret_buf[:available_size], dtype=torch.float32),
        )
        return data

def ppo_mcma(
    env_fn,
    actor_critic=MCMA_ActorCritic,
    ac_kwargs=dict(),
    steps_per_epoch=None,
    epochs=None,
    gamma=None,
    clip_ratio=None,
    pi_lr=None,
    vf_lr=None,
    train_pi_iters=None,
    train_v_iters=None,
    lam=None,
    max_ep_len=None,
    target_kl=None,
    save_freq=None,
    device=None,
    save_dir="ppo_training_results"
):
    # Use config values if not provided
    steps_per_epoch = steps_per_epoch or config.PPO['steps_per_epoch']
    epochs = epochs or config.PPO['epochs']
    gamma = gamma or config.PPO['gamma']
    clip_ratio = clip_ratio or config.PPO['clip_ratio']
    pi_lr = pi_lr or config.PPO['pi_lr']
    vf_lr = vf_lr or config.PPO['vf_lr']
    train_pi_iters = train_pi_iters or config.PPO['train_pi_iters']
    train_v_iters = train_v_iters or config.PPO['train_v_iters']
    lam = lam or config.PPO['lam']
    max_ep_len = max_ep_len or config.PPO['max_ep_len']
    target_kl = target_kl or config.PPO['target_kl']
    save_freq = save_freq or config.PPO['save_freq']
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create directories for saving results
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "plots"), exist_ok=True)

    # Initialize visualizer
    visualizer = ORANVisualizer(
        save_dir=os.path.join(save_dir, "plots"),
        config=config
    )

    # For testing, reduce steps per epoch
    if steps_per_epoch > 100:
        steps_per_epoch = 50  # Reduce for testing
        logger.info(f"Reduced steps_per_epoch to {steps_per_epoch} for testing")

    env = env_fn()
    ac = actor_critic(None, None, **ac_kwargs).to(device)
    gat = BipartiteGAT(
        critic_features=1,
        actor_features=1,
        hidden_dim=config.GAT['hidden_dim'],
        num_heads=config.GAT['num_heads'],
        num_critics=2,
        num_actors=2,
        dropout=config.GAT['dropout']
    ).to(device)
    buf = MCMABuffer(
        max_ues=config.ENV['max_ues'],
        act_dim=config.ACTOR_CRITIC['ee_action_dim'] + config.ACTOR_CRITIC['ns_action_dim'],
        size=steps_per_epoch,
        gamma=gamma,
        lam=lam
    )
    ee_actor_optimizer = Adam(ac.ee_actor.parameters(), lr=pi_lr)
    ns_actor_optimizer = Adam(ac.ns_actor.parameters(), lr=pi_lr)
    ee_critic_optimizer = Adam(ac.ee_critic.parameters(), lr=vf_lr)
    ns_critic_optimizer = Adam(ac.ns_critic.parameters(), lr=vf_lr)
    
    def compute_loss_pi(data):
        encoded = data['encoded']
        ee_act = data['ee_act']
        ns_act = data['ns_act']
        ee_adv = data['ee_adv']
        ns_adv = data['ns_adv']
        ee_logp_old = data['ee_logp']
        ns_logp_old = data['ns_logp']
        # Policy loss for EE actor
        ee_logp, ns_logp = ac.get_log_prob(encoded, ee_act, ns_act)
        ratio_ee = torch.exp(ee_logp - ee_logp_old)
        ratio_ns = torch.exp(ns_logp - ns_logp_old)
        clip_adv_ee = torch.clamp(ratio_ee, 1-clip_ratio, 1+clip_ratio) * ee_adv
        clip_adv_ns = torch.clamp(ratio_ns, 1-clip_ratio, 1+clip_ratio) * ns_adv
        loss_pi_ee = -(torch.min(ratio_ee * ee_adv, clip_adv_ee)).mean()
        loss_pi_ns = -(torch.min(ratio_ns * ns_adv, clip_adv_ns)).mean()
        # Entropy (for logging)
        ee_entropy, ns_entropy = ac.get_entropy(encoded)
        pi_info = dict(
            kl_ee=(ee_logp - ee_logp_old).mean().item(),
            kl_ns=(ns_logp - ns_logp_old).mean().item(),
            ent_ee=ee_entropy.mean().item(),
            ent_ns=ns_entropy.mean().item()
        )
        return loss_pi_ee, loss_pi_ns, pi_info
    
    def compute_loss_v(data):
        encoded = data['encoded']
        ee_ret = data['ee_ret']
        ns_ret = data['ns_ret']
        ee_v, ns_v = ac.get_values(encoded)
        ee_loss = F.smooth_l1_loss(ee_v, ee_ret, reduction='mean')
        ns_loss = F.smooth_l1_loss(ns_v, ns_ret, reduction='mean')
        return ee_loss, ns_loss
    
    def save_model(epoch):
        model_path = os.path.join(save_dir, "models", f'mcma_ppo_epoch_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'model_state_dict': ac.state_dict(),
            'config': config,
            'metrics': visualizer.metrics
        }, model_path)
        logger.info(f"Model saved to: {model_path}")
    
    start_time = time.time()
    o, ep_ret_ee, ep_ret_ns, ep_len = env.reset(), 0, 0, 0
    
    # Enhanced epoch stats for visualization
    epoch_stats = {
        'ep_ret_ee': [],
        'ep_ret_ns': [],
        'ep_len': [],
        'ee_loss': [],
        'ns_loss': [],
        'ee_actions': [],
        'ns_actions': [],
        'pi_loss_ee': [],
        'pi_loss_ns': [],
        'kl_div_ee': [],
        'kl_div_ns': [],
        'entropy_ee': [],
        'entropy_ns': [],
        'clip_frac_ee': [],
        'clip_frac_ns': []
    }
    
    def update():
        data = buf.get()
        
        # Policy updates
        for i in range(train_pi_iters):
            ee_actor_optimizer.zero_grad()
            loss_pi_ee, _, pi_info = compute_loss_pi(data)
            loss_pi_ee.backward()
            ee_actor_optimizer.step()
            if pi_info['kl_ee'] > 1.5 * target_kl:
                logger.info(f'Early stopping EE actor at iteration {i} due to reaching max kl')
                break
        
        for i in range(train_pi_iters):
            ns_actor_optimizer.zero_grad()
            _, loss_pi_ns, pi_info = compute_loss_pi(data)
            loss_pi_ns.backward()
            ns_actor_optimizer.step()
            if pi_info['kl_ns'] > 1.5 * target_kl:
                logger.info(f'Early stopping NS actor at iteration {i} due to reaching max kl')
                break
        
        # Value function updates
        for i in range(train_v_iters):
            ee_critic_optimizer.zero_grad()
            loss_v_ee, _ = compute_loss_v(data)
            loss_v_ee.backward()
            torch.nn.utils.clip_grad_norm_(ac.ee_critic.parameters(), max_norm=0.5)
            ee_critic_optimizer.step()
            epoch_stats['ee_loss'].append(loss_v_ee.item())
        
        for i in range(train_v_iters):
            ns_critic_optimizer.zero_grad()
            _, loss_v_ns = compute_loss_v(data)
            loss_v_ns.backward()
            torch.nn.utils.clip_grad_norm_(ac.ns_critic.parameters(), max_norm=0.5)
            ns_critic_optimizer.step()
            epoch_stats['ns_loss'].append(loss_v_ns.item())
    
    # Main training loop
    for epoch in range(epochs):
        epoch_start_time = time.time()
        
        for t in range(steps_per_epoch):
            try:
                # Get encoded state and actions/values/logps
                ac_out = ac.step(o)
                encoded = ac_out['encoded'].cpu().numpy()
                # EE actions should be integers (discrete actions) - ensure proper conversion
                ee_act_tensor = ac_out['ee_action'].cpu()
                ee_act = ee_act_tensor.numpy().astype(np.int32)  # Ensure they remain integers
                ns_act = ac_out['ns_action'].cpu().numpy().astype(np.float32)  # Continuous actions as float32
                ee_logp = ac_out['ee_logp'].cpu().numpy()
                ns_logp = ac_out['ns_logp'].cpu().numpy()
                ee_val = ac_out['ee_value'].cpu().numpy()
                ns_val = ac_out['ns_value'].cpu().numpy()
                
                # Debug: Verify action types
                print(f"EE actions tensor dtype: {ee_act_tensor.dtype}, values: {ee_act_tensor.numpy()}")
                print(f"EE actions numpy dtype: {ee_act.dtype}, values: {ee_act}")
                print(f"NS actions numpy dtype: {ns_act.dtype}, values: {ns_act}")
                
                # GAT aggregation (for each actor)
                ee_agg_val, ns_agg_val = gat([
                    torch.tensor(ee_val).reshape(1),
                    torch.tensor(ns_val).reshape(1)
                ])
                ee_agg_val = ee_agg_val.cpu().detach().numpy().squeeze()
                ns_agg_val = ns_agg_val.cpu().detach().numpy().squeeze()
                
                # Step env with combined actions: [slice1, slice2, slice3, a_t, b_t, c_t]
                action = np.concatenate([ns_act, ee_act])  # Total 6 actions
                o2, (r_ee, r_ns), d, info = env.step(action)
                ep_ret_ee += r_ee
                ep_ret_ns += r_ns
                ep_len += 1
                
                # Store actions for visualization
                epoch_stats['ee_actions'].append(ee_act)
                epoch_stats['ns_actions'].append(ns_act)
                
                # Store in buffer (even if using default state)
                buf.store(
                    o, encoded, ee_act, ns_act, ee_logp, ns_logp, ee_val, ns_val, ee_agg_val, ns_agg_val, r_ee, r_ns
                )
                o = o2
                
                if d or (ep_len == max_ep_len):
                    epoch_stats['ep_ret_ee'].append(ep_ret_ee)
                    epoch_stats['ep_ret_ns'].append(ep_ret_ns)
                    epoch_stats['ep_len'].append(ep_len)
                    # Finish the path to compute advantages
                    buf.finish_path()
                    o, ep_ret_ee, ep_ret_ns, ep_len = env.reset(), 0, 0, 0
                    
            except Exception as e:
                logger.error(f"Error in training step {t}: {e}")
                # Skip this step and continue
                continue
        
        # Check if buffer is ready for update
        if buf.ptr >= buf.max_size:
            if (epoch % save_freq == 0) or (epoch == epochs-1):
                save_model(epoch)
            update()
        else:
            logger.info(f"Buffer not full yet ({buf.ptr}/{buf.max_size}), skipping update for epoch {epoch}")
            continue
        
        # Calculate epoch metrics for visualization
        avg_ep_ret_ee = np.mean(epoch_stats["ep_ret_ee"]) if epoch_stats["ep_ret_ee"] else 0
        avg_ep_ret_ns = np.mean(epoch_stats["ep_ret_ns"]) if epoch_stats["ep_ret_ns"] else 0
        avg_ep_len = np.mean(epoch_stats["ep_len"]) if epoch_stats["ep_len"] else 0
        avg_ee_loss = np.mean(epoch_stats["ee_loss"]) if epoch_stats["ee_loss"] else 0
        avg_ns_loss = np.mean(epoch_stats["ns_loss"]) if epoch_stats["ns_loss"] else 0
        
        # Calculate action metrics
        ee_actions_array = np.array(epoch_stats['ee_actions']) if epoch_stats['ee_actions'] else np.array([])
        ns_actions_array = np.array(epoch_stats['ns_actions']) if epoch_stats['ns_actions'] else np.array([])
        
        # Prepare metrics for visualizer
        training_metrics = {
            'ep_ret': (avg_ep_ret_ee + avg_ep_ret_ns) / 2,  # Combined return
            'ep_len': avg_ep_len,
            'ep_ee_reward': avg_ep_ret_ee,
            'ep_ns_reward': avg_ep_ret_ns,
            'ep_total_reward': avg_ep_ret_ee + avg_ep_ret_ns,
            'pi_loss': (avg_ee_loss + avg_ns_loss) / 2,  # Combined policy loss
            'v_loss': (avg_ee_loss + avg_ns_loss) / 2,   # Combined value loss
            'kl_div': 0.0,  # Will be updated if available
            'entropy': 0.0,  # Will be updated if available
            'clip_frac': 0.0,  # Will be updated if available
            'explained_var': 0.0  # Will be updated if available
        }
        
        # Calculate network metrics (placeholder - should be implemented based on your environment)
        network_metrics = calculate_network_metrics(
            env_state=info if 'info' in locals() else {},
            actions={'ee': ee_actions_array, 'ns': ns_actions_array}
        )
        
        # Calculate action metrics
        action_metrics = calculate_action_metrics(
            ee_actions=ee_actions_array,
            ns_actions=ns_actions_array
        )
        
        # Calculate multi-objective metrics
        try:
            multi_obj_metrics = calculate_multi_objective_metrics(
                ee_reward=avg_ep_ret_ee,
                ns_reward=avg_ep_ret_ns
            )
        except Exception as e:
            logger.warning(f"Error calculating multi-objective metrics: {e}")
            multi_obj_metrics = {
                'pareto_efficiency': 0.0,
                'objective_conflicts': 1.0,
                'fairness_index': 1.0
            }
        
        # Update visualizer
        try:
            visualizer.update_metrics(
                epoch=epoch,
                training_metrics={**training_metrics, **multi_obj_metrics},
                network_metrics=network_metrics,
                action_metrics=action_metrics
            )
            
            # Print summary
            visualizer.print_summary(epoch)
            
            # Generate plots periodically
            visualizer.generate_all_plots(epoch, save=True, show=False)
        except Exception as e:
            logger.warning(f"Error in visualization for epoch {epoch}: {e}")
            # Continue training even if visualization fails
        
        # Logging
        logger.info(f'Epoch {epoch}')
        logger.info(f'Average EpRetEE: {avg_ep_ret_ee:.3f}')
        logger.info(f'Average EpRetNS: {avg_ep_ret_ns:.3f}')
        logger.info(f'Average EpLen: {avg_ep_len:.3f}')
        logger.info(f'Average EE Loss: {avg_ee_loss:.3f}')
        logger.info(f'Average NS Loss: {avg_ns_loss:.3f}')
        logger.info(f'TotalEnvInteracts: {(epoch+1)*steps_per_epoch}')
        logger.info(f'Time: {time.time()-start_time:.3f}')
        
        # Save metrics periodically
        if (epoch + 1) % save_freq == 0:
            visualizer.save_metrics(f"metrics_epoch_{epoch}.json")
        
        # Reset epoch stats
        epoch_stats = {
            'ep_ret_ee': [],
            'ep_ret_ns': [],
            'ep_len': [],
            'ee_loss': [],
            'ns_loss': [],
            'ee_actions': [],
            'ns_actions': [],
            'pi_loss_ee': [],
            'pi_loss_ns': [],
            'kl_div_ee': [],
            'kl_div_ns': [],
            'entropy_ee': [],
            'entropy_ns': [],
            'clip_frac_ee': [],
            'clip_frac_ns': []
        }
    
    # Final save and visualization
    try:
        visualizer.save_metrics("final_metrics.json")
        visualizer.create_dashboard(save_html=True)
        logger.info(f"Training completed. Results saved to: {save_dir}")
    except Exception as e:
        logger.warning(f"Error in final visualization save: {e}")
        logger.info(f"Training completed. Results saved to: {save_dir}")

def main():
    """
    Main entry point for MCMA-PPO training
    Following SpinningUp argument parsing structure
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='OranEnv')
    parser.add_argument('--exp_name', type=str, default='mcma-ppo')
    parser.add_argument('--save_dir', type=str, default='ppo_training_results')
    args = parser.parse_args()
    
    # Environment configuration
    def env_fn():
        return OranEnv(
            num_slices=config.ENV['num_slices'],
            N_sf=config.ENV['N_sf'],
            user_num=config.ENV['user_num']
        )
    
    # Model configuration - MCMA_ActorCritic doesn't need these parameters
    ac_kwargs = dict(
        hidden_sizes=config.ACTOR_CRITIC['hidden_sizes']
    )
    
    # Create save directory with experiment name
    save_dir = os.path.join(args.save_dir, args.exp_name)
    
    # Run PPO
    ppo_mcma(
        env_fn=env_fn,
        actor_critic=MCMA_ActorCritic,
        ac_kwargs=ac_kwargs,
        save_dir=save_dir
    )

if __name__ == '__main__':
    main() 