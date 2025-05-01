import torch
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
from math import pi
from collections import deque
from network import Actor, Critic, ReplayBuffer
from phase_setting import get_phase_config
import copy

def init_display_settings():
    np.set_printoptions(
        precision=8,                     # 控制小數點顯示的位數
        suppress=True,                   # 不使用科學計數法
        floatmode='fixed',               # 顯示固定小數位數
        threshold=np.inf,                # 陣列中元素的顯示數量
        edgeitems=30,                    # 當陣列太大需要省略顯示時, 控制在開始和結尾處顯示的元素數量
        linewidth = 1000000              # 控制輸出時每行的最大字元數, 避免換行
    )
    th.set_printoptions(precision=8)
    pd.set_option('display.precision', 8)
    pd.set_option('display.float_format', lambda x: '%.8f' % x)

def gumbel_softmax_sample(logits, tau=1.0, eps=1e-20):
    U = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(U + eps) + eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)

class EMARewardScaler:
    def __init__(self, args):
        self.ema_mean = 0
        self.ema_var = 1
        self.beta = args.ema_beta
        self.step = 0

    def normalize(self, reward):
        self.step += 1
        self.ema_mean = self.beta * self.ema_mean + (1 - self.beta) * reward
        mean = self.ema_mean / (1 - self.beta ** self.step)

        self.ema_var = self.beta * self.ema_var + (1 - self.beta) * ((reward - mean) ** 2)
        std = (self.ema_var / (1 - self.beta ** self.step)) ** 0.5

        return (reward - mean) / (std + 1e-8)


    # def __init__(self, beta=0.99):
    #     self.ema_mean = 0
    #     self.ema_sq_mean = 0
    #     self.beta = beta
    #     self.step = 0

    # def normalize(self, reward):
    #     self.step += 1
    #     self.ema_mean = self.beta * self.ema_mean + (1 - self.beta) * reward
    #     self.ema_sq_mean = self.beta * self.ema_sq_mean + (1 - self.beta) * (reward ** 2)

    #     unbiased_mean = self.ema_mean / (1 - self.beta ** self.step)
    #     unbiased_sq_mean = self.ema_sq_mean / (1 - self.beta ** self.step)

    #     variance = unbiased_sq_mean - unbiased_mean ** 2
    #     std = np.sqrt(max(variance, 1e-6))  # 避免 sqrt(負數)

    #     return (reward - unbiased_mean) / (std + 1e-8)


    # """ 使用指數移動平均 (EMA) 來標準化 Reward, 使其自適應變化 """
    # def __init__(self, beta=0.99):
    #     self.ema_mean = 0
    #     self.ema_std = 1
    #     self.beta = beta  # 控制平滑程度（越接近 1，變化越慢）
    #     self.step = 0

    # def normalize(self, reward):
    #     self.step += 1
    #     self.ema_mean = self.beta * self.ema_mean + (1 - self.beta) * reward
    #     unbiased_mean = self.ema_mean / (1 - self.beta ** self.step)
    #     return reward / (unbiased_mean + 1e-8)

class DDPGAgent:
    def __init__(self, K, state_dim, num_groups, num_phases, neuron, args):
        self.device = args.device
        self.gamma = args.gamma
        self.tau = args.tau

        # 在 DDPG 內, Critic 需要比 Actor 更新更快, 如果 Critic 太弱, Actor 學不到真正的 Reward 差異
        self.policy_delay = args.policy_delay  # 每間隔 policy_delay 次更新 Actor
        self.exploration_rate = args.exploration_rate   # 初始 100% 探索
        self.exploration_decay = args.exploration_decay  # 每次訓練後探索率乘上這個值
        self.exploration_min = args.exploration_min     # 最低探索率
        self.total_steps = 0  # 如果 `select_action()` 會用到，就保留；如果沒用到，就刪除

        self.batch_size = args.batch_size
        self.num_groups = num_groups    # 群組數量
        self.num_phases = num_phases    # 2-bit phase (4 個選項)

        # 設定 Actor、Critic 和 Optimizer
        self.actor = Actor(state_dim, num_groups, num_phases, neuron).to(self.device)
        self.critic = Critic(state_dim, num_groups, num_phases, neuron).to(self.device)
        self.target_actor = Actor(state_dim, num_groups, num_phases, neuron).to(self.device)
        self.target_critic = Critic(state_dim, num_groups, num_phases, neuron).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.eval()
        self.target_critic.eval()

        # 目標網路初始化為當前網路
        self.update_target_networks(tau=1.0)

        # self.target_actor.load_state_dict(self.actor.state_dict())
        # self.target_critic.load_state_dict(self.critic.state_dict())

        # 設定 Adam Optimizer
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.loss_fn = nn.SmoothL1Loss()
        self.critic_loss = None
        self.actor_loss = None

        # 記憶體 (Replay Buffer)
        self.memory = ReplayBuffer(args.buffer_size, K, state_dim, num_groups, self.device)

        # 取得離散 phase 設定
        self.num_phases, self.phases_discrete = get_phase_config(args)

        # 動態 Reward 標準化
        self.ema_scaler = EMARewardScaler(args)
        self.gumbel_tau = args.gumbel_tau

    def select_action(self, state, num_elements, eval=False):
        # print(f"agent.py/DDPGAgent/select_action || state: {state}")
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        # print(f"agent.py/DDPGAgent/select_action || state: {state}")

        logits = self.actor(state).squeeze(0)  # shape: (num_groups, num_phases), 對 n 個 RIS group 輸出兩個動作(phase)的機率分布
        # print(f"agent.py/DDPGAgent/select_action || logits: {logits.shape} \n{logits}")
        
        if eval:
            action = logits.argmax(dim=-1).cpu().numpy()
            # action_probs = self.actor(state).detach().cpu().numpy().squeeze(0)  # shape: (num_groups, num_phases)
            # # print(f"agent.py/DDPGAgent/select_action || action_probs: {action_probs.shape} \n{action_probs}")
        else:
            if np.random.rand() < self.exploration_rate:
                # 探索: 隨機選擇每個群組的 phase
                action = np.random.randint(0, self.num_phases, size=(self.num_groups,))
                print(f"agent.py/DDPGAgent/select_action || random action: {action}\n")
            else:
                # Gumbel-Softmax 選擇行動
                # 使用 Gumbel-Softmax 根據 logits 機率隨機選出一個動作（加入隨機性但仍可訓練）
                # hard=True 代表每組只會選出一個機率最大的動作（像是擲骰子後選一個結果）
                # Gumbel-Softmax 用來加入隨機性選擇動作（可訓練）
                # 即使原本第一項機率較高，加入 Gumbel noise 後，第二項也可能被選中
                # 例如原本 [0.8, 0.2]，加入 noise 後可能變成 [0.6, 0.7] → 選第二項

                y = gumbel_softmax_sample(logits, self.gumbel_tau)  # shape: (num_groups, num_phases)
                action = y.argmax(dim=-1).cpu().numpy()
                print(f"agent.py/DDPGAgent/select_action || Gumbel-softmax action: {action}\n")

                # # 根據機率最大值選擇 phase
                # action = np.argmax(action_probs, axis=-1)
                # print(f"agent.py/DDPGAgent/select_action || argmax action: {action}\n")

        return action

    def sample_priority_replay(self):
        """ 讓高 Datarate 的 sample 更容易被選到 """
        sinr_values = np.array([s[2].cpu().item() if isinstance(s[2], torch.Tensor) else s[2] for s in self.memory], dtype=np.float32).flatten()
        # print(f"agent.py/DDPGAgent/sample_priority_replay || sinr_values: {sinr_values}")
        prob = sinr_values / sinr_values.sum()
        prob = prob.flatten()  # 確保 `probabilities` 是 1D 陣列
        # print(f"agent.py/DDPGAgent/sample_priority_replay || prob: {prob}")
        indices = np.random.choice(len(self.memory), size=self.batch_size, replace=False, p=prob)
        # print(f"agent.py/DDPGAgent/sample_priority_replay || indices: {indices}\n")
        output = [self.memory[i] for i in indices]
        # print(f'output: {output}')
        return output

    def train(self, episode, temp_step):
        init_display_settings()

        if len(self.memory) < self.batch_size:
            return

        # # DEBUG: 從 `self.memory` 取出傳送過來的 next_state 來確認
        # # for i, transition in enumerate(list(self.memory)[:10]):  # 只遍歷前 10 筆
        # #     print(f"agent.py/DDPGAgent/train || Transition {i}: next_state = {transition[3]}")  # `self.memory` 中的 `next_state` 是 tuple 的第 4 個元素
        # # batch = self.sample_priority_replay()       # 讓較高 datarate 的 sample 被選到
        # batch = random.sample(self.memory, self.batch_size)       # orig: 隨機抽樣
        # # print(f"agent.py/DDPGAgent/train || batch: {batch}")

        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # print(f"agent.py/DDPGAgent/train || state: {state}")
        # print(f"agent.py/DDPGAgent/train || action: {action}")
        # print(f"agent.py/DDPGAgent/train || reward: {reward}")
        # print(f"agent.py/DDPGAgent/train || next_state: {next_state}")
        # print(f"agent.py/DDPGAgent/train || done: {done}")

        # 把 actions 轉成 NumPy 陣列，避免 tuple 轉換錯誤
        states = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in state], dtype=np.float32)
        # print(f"agent.py/DDPGAgent/train || states: {states}")
        actions = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in action], dtype=np.float32)
        # print(f"agent.py/DDPGAgent/train || actions: {actions}")
        rewards = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in reward], dtype=np.float32)
        # print(f"agent.py/DDPGAgent/train || rewards: {rewards}")
        next_states = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in next_state], dtype=np.float32)
        # print(f"agent.py/DDPGAgent/train || next_states: {next_states}")
        dones = np.array([x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in done], dtype=np.float32)
        # print(f"agent.py/DDPGAgent/train || dones: {dones}\n")

        # 標準化 Reward
        # normalized_rewards = torch.tensor(
        #     [self.ema_scaler.normalize(float(r)) for r in rewards], dtype=torch.float32, device=self.device
        # )
        normalized_rewards = torch.tensor(
            [self.ema_scaler.normalize(float(val)) for r in rewards for val in np.atleast_1d(r)],
            dtype=torch.float32, device=self.device
        )
        # # 更新 EMA 參數
        # for r in rewards:
        #     self.ema_scaler.normalize(float(r))

        # 轉換為 Tensor
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        # print(f"agent.py/DDPGAgent/train || states: {states}")
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)  # 保持維度一致
        # print(f"agent.py/DDPGAgent/train || actions: {actions}")
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        # print(f"agent.py/DDPGAgent/train || next_states: {next_states}")
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        # print(f"agent.py/DDPGAgent/train || dones: {dones}")

        # Critic 更新
        with torch.no_grad():
            next_action_probs = self.target_actor(next_states)  
            next_actions = torch.argmax(next_action_probs, dim=-1, keepdim=True)  # 確保維度一致
            # print(f"agent.py/DDPGAgent/train || next_actions: {next_actions}")
            target_Q = self.target_critic(next_states, next_actions).detach()
            target_value = normalized_rewards + (1 - dones) * self.gamma * target_Q
            # print(f"agent.py/DDPGAgent/train || target_value: {target_value}")

        current_Q = self.critic(states, actions)

        self.critic_loss = self.loss_fn(current_Q, target_value)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()

        # Actor 更新 (每 `policy_delay` 次更新一次)
        if self.total_steps % self.policy_delay == 0:
            logits = self.actor(states)  # shape: (batch, num_groups, num_phases)
            soft_actions = gumbel_softmax_sample(logits, self.gumbel_tau)  # 可導
            # 將 soft one-hot argmax (浮點數) 轉為 index (整數)
            # action_indices = soft_actions.argmax(dim=-1, keepdim=True)

            # action_probs = self.actor(states)  
            # selected_actions = torch.argmax(action_probs, dim=-1, keepdim=True)  

            actor_loss = -self.critic.forward_with_soft_action(states, soft_actions).mean()
            # actor_loss = -self.critic(states, selected_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.update_target_networks()

            self.actor_loss = actor_loss.item()

        self.total_steps += 1

        # 讓前 150 episode 探索率降得更慢
        if episode < 200:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * 0.99)
        else:
            self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)
        print(f"Updated Exploration Rate: {self.exploration_rate:.4f}")

        # Gumbel Softmax 的溫度逐步降低
        self.gumbel_tau = max(0.05, self.gumbel_tau * 0.99995)
        # self.gumbel_tau = max(0.1, self.gumbel_tau * 0.995)
        print(f"[Debug] Gumbel Tau: {self.gumbel_tau:.4f}")

        # === [DEBUG] Q值與Loss的偵錯資訊 ===
        print(f"[Debug] Normalized Rewards: min={normalized_rewards.min().item():.4f}, max={normalized_rewards.max().item():.4f}, mean={normalized_rewards.mean().item():.4f}")
        print(f"[Debug] Q-values: current_Q mean={current_Q.mean().item():.4f}, target_Q mean={target_Q.mean().item():.4f}")
        print(f"[Debug] Q diff mean abs: {(target_Q - current_Q).abs().mean().item():.6f}")
        print(f"[Debug] Critic Loss: {self.critic_loss.item():.8f}")

        # 每 5000 step 繪製一次 Q 值圖
        if self.total_steps % 5000 == 0:
            import matplotlib.pyplot as plt
            current = current_Q.detach().cpu().numpy().flatten()
            target = target_Q.detach().cpu().numpy().flatten()
            plt.figure(figsize=(8, 4))
            plt.plot(current, label='Current Q')
            plt.plot(target, label='Target Q')
            plt.title(f'Q Value Comparison at Step {self.total_steps}')
            plt.xlabel('Sample Index')
            plt.ylabel('Q Value')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f'./q_value_plot_step/q_value_plot_step{self.total_steps}.png')
            plt.close()

        #==================================

    def update_target_networks(self, tau=None):
        tau = self.tau if tau is None else tau  
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def getLoss(self):
        return self.critic_loss, self.actor_loss

    def getCheckpoint(self):
        """
        返回 DDPG 模型的 checkpoint, 包含 Actor, Critic 以及 Optimizer 狀態。
        """
        return {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }

