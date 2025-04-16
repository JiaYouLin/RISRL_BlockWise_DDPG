import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random

def init_display_settings():
    np.set_printoptions(
        precision=8,                     # 控制小數點顯示的位數
        suppress=True,                   # 不使用科學計數法
        floatmode='fixed',               # 顯示固定小數位數
        threshold=np.inf,                # 陣列中元素的顯示數量
        edgeitems=30,                    # 當陣列太大需要省略顯示時，控制在開始和結尾處顯示的元素數量
        linewidth = 1000000              # 控制輸出時每行的最大字元數，避免換行
    )
    th.set_printoptions(precision=8)
    pd.set_option('display.precision', 8)
    pd.set_option('display.float_format', lambda x: '%.8f' % x)

class Actor(nn.Module):
    def __init__(self, state_dim, num_elements, num_actions):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_elements * num_actions)  # 64 個 RIS element，每個有 4 個 phase shift
        self.num_elements = num_elements
        self.num_actions = num_actions

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.num_elements, self.num_actions)  # (batch, 64, 4)
        return F.softmax(x, dim=-1)  # 每個 RIS element 產生 4 個機率分佈

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PPO:
    def __init__(self, state_dim, num_elements, num_actions, args):
        init_display_settings()

        self.actor = Actor(state_dim, num_elements, num_actions).cuda()
        self.critic = Critic(state_dim).cuda()
        self.memory = []
        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.eps_clip = args.eps_clip
        self.update_freq = args.update_freq
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.lr)

        self.seed = args.seed                                                                        # 設定隨機種子
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        
        # # 設定 PhaseShift 離散值
        # self.num_phases = 2 ** self.bits  # 計算總共的 PhaseShift 數量
        # self.phases_discrete = torch.linspace(
        #     start = 0.0,                      # 確保是 float
        #     end = 2 * np.pi,                  # 確保是 float
        #     steps = self.num_phases,
        #     dtype = torch.float32,
        # ).to(self.device)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).cuda().unsqueeze(0)  # (1, state_dim)
        action_probs = self.actor(state)  # (1, 64, 4)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()  # (1, 64)
        log_prob = dist.log_prob(action)
        return action.squeeze(0).cpu().numpy(), log_prob.squeeze(0).cpu().numpy()

    def store_transition(self, state, action, log_prob, reward, next_state, done):
        self.memory.append((state, action, log_prob, reward, next_state, done))

    def update(self):
        states, actions, log_probs, rewards, next_states, dones = zip(*self.memory)
        states = torch.tensor(np.array(states), dtype=torch.float32).cuda()
        actions = torch.tensor(np.array(actions), dtype=torch.long).cuda()
        log_probs = torch.tensor(np.array(log_probs), dtype=torch.float32).cuda()
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).cuda()
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).cuda()
        dones = torch.tensor(np.array(dones), dtype=torch.float32).cuda()

        values = self.critic(states).squeeze()
        next_values = self.critic(next_states).squeeze()
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_target - values.detach()

        # Actor Loss
        new_action_probs = self.actor(states)
        new_dist = torch.distributions.Categorical(new_action_probs)
        new_log_probs = new_dist.log_prob(actions)

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic Loss
        critic_loss = F.mse_loss(values, td_target.detach())

        # Update Network
        loss = actor_loss + 0.5 * critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.memory = []


# def index_to_action(self, index, num_elements):
#     """
#     57 >> [0, 3, 2, 1]

#     action_dim: element數量, 會作為組合數量的基數, 如 4^4 次方

#     將組合 index 轉換為組合 action([1, 3, 0, 2])
#     這裡用的方式是採用進制還原取得組合 action
#     舉例:
#     num_elements = 4, 每個元素有 self.action_dim = 4 種動作, 則總共有 4^4=256 種組合。
#     index 是所有動作組合的編號(範圍是 0 到 255)。
#     若 index = 57, 對應的動作組合為:
#         將 index 視為 4 進制 數字(因為 self.action_dim = 4)。
#         若 index = 57, self.action_dim = 4, num_elements = 4
#         第一次: 57%4=1, 57//4=14, 動作 [1]
#         第二次: 14%4=2, 14//4=3, 動作 [1, 2]
#         第三次: 3%4=3, 3//4=0, 動作 [1, 2, 3]
#         第四次: 0%4=0, 動作 [1, 2, 3, 0]
#         結果為 [1, 2, 3, 0], 但需要反轉, 變成 [0, 3, 2, 1]
#     """
#     actions = []
    
#     # DEBUG
#     # index =57
#     # num_elements = 4

#     for _ in range(num_elements):
#         actions.append(index % self.action_dim)      # 取餘數，還原最右側元素的動作
#         index //= self.action_dim                    # 除以進制數，將下一位移到餘數位置
#         # print(f"agent.py/index_to_action || actions: {actions}")
    
#     # DEBUG
#     # print(f"agent.py/index_to_action || list(reversed(actions)): {list(reversed(actions))}")
    
#     return list(reversed(actions))

# def action_to_index(self, action_tensor):
#     """
#     [0, 3, 2, 1] >> 57

#     將組合動作轉換為 index
#     action_tensor: 所有組合的 Tensor
#     a_dim: element數量, 會作為組合數量的基數, 如 4^4 次方
#     """
#     # print(f"network.py/action_to_index || ction_tensor: {action_tensor}")

#     # DEBUG
#     # action_tensor = torch.tensor([0, 3, 2, 1], dtype=torch.float32)  # 確保 action_tensor 是 torch.Tensor

#     # 確保 action_tensor 是正確的形狀，並轉換為一維 list
#     if isinstance(action_tensor, torch.Tensor):
#         action_list = action_tensor.squeeze().tolist()
#     else:
#         raise ValueError("action_tensor 必須是 torch.Tensor 類型。")

#     action_index = 0
#     base = 1
#     for action in reversed(action_list):
#         action_index += int(action) * base
#         base *= self.action_dim
    
#     # DEBUG
#     # print(f"agent.py/action_to_index || action_index: {action_index}")

#     return action_index