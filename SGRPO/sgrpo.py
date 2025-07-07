import torch
import torch.optim as optim
import numpy as np
from config import config, BATCH_SIZE
from model import SGRPOPolicy
from utils import group_normalize, log_prob_ratio, kl_divergence
from state_encoder import TransformerStateEncoder
import os
import copy
from visualization import SGRPOVisualizer
from env import OranEnv
import json
from datetime import datetime
import struct
import traceback

def train_sgrpo():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        env = OranEnv()
        policy = SGRPOPolicy().to(device)
        optimizer = optim.Adam(policy.parameters(), lr=config.SGRPO['pi_lr'])
        visualizer = SGRPOVisualizer(save_dir='sgrpo_training_plots', config=config)

        steps_per_epoch = config.SGRPO['steps_per_epoch']
        epochs = config.SGRPO['epochs']
        epsilon = config.SGRPO['epsilon']
        beta_kl = config.SGRPO['beta_kl']
        target_kl = config.SGRPO['target_kl']
        save_freq = config.SGRPO['save_freq']
        G = config.SGRPO['group_size']  # Number of actions per group

        print(f"Starting SGRPO training for {epochs} epochs with {steps_per_epoch} steps per epoch")
        print(f"Configuration: G={G}, target_kl={target_kl}, epsilon={epsilon}, beta_kl={beta_kl}")

        # Initialize step-level metrics storage
        step_metrics = {
            'epoch': [],
            'step': [],
            'ep_ret': [],
            'pi_loss': [],
            'kl_div': [],
            'group_adv_mean': [],
            'group_adv_std': [],
            'policy_updated': [],
            'action_diversity_slice': [],
            'action_diversity_sleep': [],
            'timestamp': []
        }

        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Starting Epoch {epoch+1}/{epochs}")
            print(f"{'='*60}")
            
            try:
                obs_buf, action_buf, reward_buf, old_logp_buf, group_adv_buf = [], [], [], [], []
                ue_state_buf = []
                env_state = env.reset(group_size=G)
                done = False
                ep_ret, ep_len = 0, 0
                per_slice_throughput = {s: [] for s in range(config.ENV['num_slices'])}
                per_slice_delay = {s: [] for s in range(config.ENV['num_slices'])}
                qos_violations = {s: 0 for s in range(config.ENV['num_slices'])}

                # Initialize old_policy as the current policy at the start of epoch
                old_policy = copy.deepcopy(policy).to(device)
                old_policy.eval()

                last_state = env.get_all_state()  # Initial observation

                for step in range(steps_per_epoch):
                    try:
                        ue_states = last_state  # Use last state as observation
                        ue_states_torch = torch.tensor(ue_states, dtype=torch.float32, device=device).unsqueeze(0)  # [BATCH_SIZE, num_ues, num_features]
                        ue_slice_ids = env.get_user_slice_ids().to(device)  # [num_ues]

                        group_actions = []
                        group_logps = []
                        group_obs = []

                        # Get normalized QoS targets for this step
                        qos_targets_torch = env.get_normalized_qos_targets().to(device)
                        
                        # Sample G actions from the same observation using old_policy
                        for g in range(G):
                            with torch.no_grad():
                                slicing_action, sleep_action = old_policy.sample_actions(ue_states_torch, ue_slice_ids, qos_targets_torch)
                                slicing_action_np = slicing_action.detach().cpu().numpy()
                                sleep_action_np = sleep_action.detach().cpu().numpy()
                                slicing_logp_old, sleep_logp_old = old_policy.log_prob(ue_states_torch, slicing_action, sleep_action, ue_slice_ids, qos_targets_torch)
                                group_actions.append((slicing_action_np, sleep_action_np))
                                group_logps.append((slicing_logp_old.detach(), sleep_logp_old.detach()))
                                group_obs.append(ue_states)

                        # Execute group actions and get rewards using env.step
                        last_state, group_rewards, _, _ = env.step(group_actions)

                        # Add detailed logging for debugging
                        print(f"\n=== Step {step+1} Debug Info ===")
                        print(f"Group actions:")
                        for g in range(G):
                            slicing_act, sleep_act = group_actions[g]
                            print(f"  Action {g}: slice={slicing_act}, sleep={sleep_act}, reward={group_rewards[g]:.3f}")
                        print(f"Group rewards: {[f'{r:.3f}' for r in group_rewards]}")
                        print(f"Reward stats: mean={np.mean(group_rewards):.3f}, std={np.std(group_rewards):.3f}")
                        print("=" * 50)

                        # Compute group-normalized advantages for the G actions
                        group_rewards_tensor = torch.tensor(group_rewards, dtype=torch.float32, device=device)
                        mean, std, advantages = group_normalize(group_rewards_tensor)
                        
                        # Additional advantage normalization for stability (added)
                        if std > 0:
                            advantages = advantages / (std + 1e-8)  # Normalize to unit variance
                        else:
                            advantages = advantages - mean  # Just center if no variance
                        
                        # Prepare tensors for update
                        slicing_logps_old = torch.stack([x[0] for x in group_logps])  # [G]
                        sleep_logps_old = torch.stack([x[1] for x in group_logps])    # [G]
                        
                        # For the same states and actions, compute log-probs under current policy
                        slicing_logps_new = []
                        sleep_logps_new = []
                        for i in range(G):
                            ue_states = torch.tensor(group_obs[i], dtype=torch.float32, device=device).unsqueeze(0)
                            ue_slice_ids = env.get_user_slice_ids().to(device)
                            slicing_action = torch.tensor(group_actions[i][0], dtype=torch.float32, device=device)
                            sleep_action = torch.tensor(group_actions[i][1], dtype=torch.long, device=device)
                            slicing_logp_new, sleep_logp_new = policy.log_prob(ue_states, slicing_action, sleep_action, ue_slice_ids, qos_targets_torch)
                            slicing_logps_new.append(slicing_logp_new)
                            sleep_logps_new.append(sleep_logp_new)
                        slicing_logps_new = torch.stack(slicing_logps_new)  # [G]
                        sleep_logps_new = torch.stack(sleep_logps_new)      # [G]
                        
                        # Joint log-probabilities
                        joint_logp_new = slicing_logps_new + sleep_logps_new
                        joint_logp_old = slicing_logps_old + sleep_logps_old
                        
                        # Joint probability ratio
                        joint_ratio = torch.exp(joint_logp_new - joint_logp_old)
                        clip_low, clip_high = 1 - epsilon, 1 + epsilon
                        
                        # Clipped surrogate objective (joint)
                        joint_obj = torch.min(
                            joint_ratio * advantages,
                            torch.clamp(joint_ratio, clip_low, clip_high) * advantages
                        )
                        
                        # KL divergence (sum of both heads, as before)
                        slicing_kl = kl_divergence(slicing_logps_old, slicing_logps_new).mean()
                        sleep_kl = kl_divergence(sleep_logps_old, sleep_logps_new).mean()
                        total_kl = slicing_kl + sleep_kl
                        
                        # Final loss with additional regularization for stability (added)
                        entropy_coef = config.SGRPO.get('entropy_coef', 0.01)
                        entropy_bonus = entropy_coef * (slicing_logps_new.mean() + sleep_logps_new.mean())  # Encourage exploration
                        loss = -joint_obj.mean() + beta_kl * total_kl - entropy_bonus
                        
                        # Policy update (only if KL divergence is acceptable)
                        if total_kl < target_kl:
                            optimizer.zero_grad()
                            loss.backward()
                            # Add gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                            optimizer.step()
                            policy_updated = True
                            
                            # Adaptive learning rate based on KL divergence (added)
                            if total_kl > target_kl * 0.5:  # If KL is getting close to limit
                                for param_group in optimizer.param_groups:
                                    param_group['lr'] = max(param_group['lr'] * 0.95, config.SGRPO['pi_lr'] * 0.1)
                        elif total_kl < target_kl * 2.5:  # Allow updates with moderate KL
                            # Use smaller learning rate for high KL updates
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = config.SGRPO['pi_lr'] * 0.5
                            optimizer.zero_grad()
                            loss.backward()
                            # Add gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.3)
                            optimizer.step()
                            # Restore original learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = config.SGRPO['pi_lr']
                            policy_updated = True
                            print(f"Warning: Moderate KL update (KL={total_kl:.4f}) with reduced LR")
                        elif total_kl < target_kl * 5:  # Allow updates even with high KL for escape
                            # Use much smaller learning rate for very high KL updates
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = config.SGRPO['pi_lr'] * 0.1
                            optimizer.zero_grad()
                            loss.backward()
                            # Add gradient clipping for stability
                            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.1)
                            optimizer.step()
                            # Restore original learning rate
                            for param_group in optimizer.param_groups:
                                param_group['lr'] = config.SGRPO['pi_lr']
                            policy_updated = True
                            print(f"Warning: High KL update (KL={total_kl:.4f}) with much reduced LR")
                        else:
                            policy_updated = False
                            print(f"Warning: Extremely high KL (KL={total_kl:.4f}), skipping update")
                            
                            # Early stopping if KL is consistently too high (added)
                            if step > 10 and total_kl > target_kl * 10:
                                print(f"Early stopping due to consistently high KL divergence")
                                break
                        
                        # SGRPO-specific logging
                        update_status = "UPDATED" if policy_updated else "SKIPPED"
                        
                        # Calculate action diversity for debugging
                        slicing_actions_array = np.array([action[0] for action in group_actions])
                        sleep_actions_array = np.array([action[1] for action in group_actions])
                        slicing_std = np.std(slicing_actions_array, axis=0).mean()
                        sleep_std = np.std(sleep_actions_array, axis=0).mean()
                        
                        print(f"[SGRPO] Epoch {epoch+1} | Step {step+1} | Loss: {loss.item():.4f} | KL: {total_kl.item():.4f} | Return: {np.mean(group_rewards):.2f} | GroupAdv mean: {mean.item():.4f} std: {std.item():.4f} | Policy: {update_status} | Action_std: slice={slicing_std:.2f} sleep={sleep_std:.2f}")
                        
                        # Store metrics for this step
                        training_metrics = {
                            'ep_ret': float(np.mean(group_rewards)),
                            'ep_len': int(G),
                            'pi_loss': float(loss.item()),
                            'kl_div': float(total_kl.item()),
                            'entropy': float((slicing_logps_new.mean() + sleep_logps_new.mean()).item()),
                            'clip_frac': float(((joint_ratio > clip_high) | (joint_ratio < clip_low)).float().mean().item()),
                            'group_adv_mean': float(mean.item()),
                            'group_adv_std': float(std.item()),
                            'policy_updated': policy_updated,
                            'action_diversity_slice': float(slicing_std),
                            'action_diversity_sleep': float(sleep_std),
                        }
                        
                        # Network performance metrics
                        slice_names = ['embb', 'urllc', 'mmtc']
                        network_metrics = {}
                        for s in range(config.ENV['num_slices']):
                            if per_slice_throughput[s]:
                                network_metrics[f'throughput_{slice_names[s]}'] = np.mean(per_slice_throughput[s])
                                network_metrics[f'delay_{slice_names[s]}'] = np.mean(per_slice_delay[s])
                                network_metrics[f'qos_violation_{slice_names[s]}'] = qos_violations[s] / max(1, len(per_slice_throughput[s]))
                        
                        # Save per-step metrics
                        step_metrics['epoch'].append(epoch)
                        step_metrics['step'].append(step)
                        step_metrics['ep_ret'].append(training_metrics['ep_ret'])
                        step_metrics['pi_loss'].append(training_metrics['pi_loss'])
                        step_metrics['kl_div'].append(training_metrics['kl_div'])
                        step_metrics['group_adv_mean'].append(training_metrics['group_adv_mean'])
                        step_metrics['group_adv_std'].append(training_metrics['group_adv_std'])
                        step_metrics['policy_updated'].append(training_metrics['policy_updated'])
                        step_metrics['action_diversity_slice'].append(training_metrics['action_diversity_slice'])
                        step_metrics['action_diversity_sleep'].append(training_metrics['action_diversity_sleep'])
                        step_metrics['timestamp'].append(datetime.now().isoformat())
                        
                        # Save step metrics to file every 10 steps
                        if (step + 1) % 10 == 0:
                            step_metrics_path = os.path.join('sgrpo_training_plots', f'step_metrics_epoch_{epoch+1}.json')
                            with open(step_metrics_path, 'w') as f:
                                json.dump(step_metrics, f, indent=2)
                            print(f"Step metrics saved: {step_metrics_path}")
                        
                    except Exception as step_error:
                        print(f"Error in step {step+1} of epoch {epoch+1}: {step_error}")
                        traceback.print_exc()
                        continue  # Continue to next step instead of stopping

                # Store action metrics once per epoch (at the end)
                action_metrics = {
                    'slicing_actions': group_actions[-1][0],  # Use the last action from the last group
                    'sleep_actions': group_actions[-1][1],    # Use the last action from the last group
                }
                
                # Record metrics ONCE per epoch (not per step)
                visualizer.update_metrics(epoch, training_metrics, network_metrics, action_metrics)
                
                # Generate plots and save metrics at the end of the epoch
                visualizer.save_metrics()
                visualizer.generate_all_plots(epoch)
                visualizer.print_summary(epoch)

                # PPO-style reference model update: update old_policy at the end of epoch
                old_policy = copy.deepcopy(policy).to(device)
                old_policy.eval()
                print(f"Updated reference model (old_policy) at end of epoch {epoch+1}")

                # Save comprehensive model checkpoint at the end of the epoch
                if (epoch + 1) % save_freq == 0:
                    model_path = os.path.join('sgrpo_training_plots', f'policy_epoch{epoch+1}.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_metrics': training_metrics,
                        'network_metrics': network_metrics,
                        'config': config.__dict__,
                        'loss': loss.item(),
                        'kl_div': total_kl.item(),
                        'ep_ret': np.mean(group_rewards),
                    }, model_path)
                    latest_model_path = os.path.join('sgrpo_training_plots', 'latest_policy.pt')
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': policy.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'training_metrics': training_metrics,
                        'network_metrics': network_metrics,
                        'config': config.__dict__,
                        'loss': loss.item(),
                        'kl_div': total_kl.item(),
                        'ep_ret': np.mean(group_rewards),
                    }, latest_model_path)
                    print(f"Model checkpoint saved: {model_path}")
                    print(f"Latest model saved: {latest_model_path}")
                    epoch_metrics = {
                        'epoch': epoch + 1,
                        'timestamp': datetime.now().isoformat(),
                        'training_metrics': training_metrics,
                        'network_metrics': network_metrics,
                        'action_metrics': action_metrics,
                        'config': config.__dict__
                    }
                    epoch_metrics_path = os.path.join('sgrpo_training_plots', f'epoch_{epoch+1}_metrics.json')
                    with open(epoch_metrics_path, 'w') as f:
                        json.dump(epoch_metrics, f, indent=2)
                    print(f"Epoch metrics saved: {epoch_metrics_path}")
                
                if (epoch + 1) % 10 == 0:
                    print(f"\n{'='*60}")
                    print(f"Detailed Progress - Epoch {epoch+1}")
                    print(f"{'='*60}")
                    print(f"Training Loss: {loss.item():.6f}")
                    print(f"KL Divergence: {total_kl.item():.6f}")
                    print(f"Episode Return: {np.mean(group_rewards):.4f}")
                    print(f"Group Advantage - Mean: {mean.item():.4f}, Std: {std.item():.4f}")
                    print(f"Energy Efficiency: {training_metrics.get('energy_efficiency', 'N/A')}")
                    print(f"Constraint Satisfaction: {training_metrics.get('constraint_satisfaction', 'N/A')}")
                    print(f"{'='*60}\n")
                
                print(f"Completed Epoch {epoch+1}/{epochs}")
                
            except Exception as epoch_error:
                print(f"Error in epoch {epoch+1}: {epoch_error}")
                traceback.print_exc()
                continue  # Continue to next epoch instead of stopping
        
        # Save final step metrics
        final_step_metrics_path = os.path.join('sgrpo_training_plots', 'final_step_metrics.json')
        with open(final_step_metrics_path, 'w') as f:
            json.dump(step_metrics, f, indent=2)
        print(f"Final step metrics saved: {final_step_metrics_path}")
        
        print(f"\n{'='*60}")
        print(f"Training completed successfully! Total epochs: {epochs}")
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"Critical error in training: {e}")
        traceback.print_exc()
        raise

if __name__ == '__main__':
    train_sgrpo() 