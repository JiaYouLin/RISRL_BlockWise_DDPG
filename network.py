#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# networtk.py

import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from math import pi
from phase_setting import get_phase_config

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

# neuron = 64
network_layer = "組合 2: 多層、同神經元數"

class Actor(nn.Module):

    def __init__(self, state_dim, num_groups, num_phases, neuron):
        """
        Actor:
        - 輸入 state
        - 輸出 `group_index` (選哪個 group)
        - 輸出 `group_phase` (該 group 內 elements 設定的 phase, 離散的 phase)
        """

        if network_layer == "Single USER (K=1)":
        
            # ==== Single USER (K=1) ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases

            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, num_groups * num_phases)  # 輸出 num_groups * 2^bits 個值
            self.softmax = nn.Softmax(dim=-1)  # 讓每個群組的 phase 選擇是機率分佈
        
        elif network_layer == "Multi USER (K>1) - 少層同神經元1":
        
            # ==== Multi USER (K>1) - 少層同神經元1 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron // 2)
            self.fc4 = nn.Linear(neuron // 2, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "Multi USER (K>1) - 少層同神經元2":

            # ==== Multi USER (K>1) - 少層同神經元2 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron // 2)
            self.fc4 = nn.Linear(neuron // 2, neuron // 2)
            self.fc5 = nn.Linear(neuron // 2, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 1: 少層、同神經元數":

            # ==== 組合 1: 少層、同神經元數 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 2: 多層、同神經元數":

            # ==== 組合 2: 多層、同神經元數 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            self.fc4 = nn.Linear(neuron, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 3: 多層、不同神經元數":

            # ==== 組合 3: 多層、不同神經元數 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron // 2)
            self.fc3 = nn.Linear(neuron // 2, neuron // 4)
            self.fc4 = nn.Linear(neuron // 4, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道":

            # ==== 組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron // 2)
            self.fc3 = nn.Linear(neuron // 2, neuron // 4)
            self.fc4 = nn.Linear(neuron // 4, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block)":

            # ==== 組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block) ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            self.out = nn.Linear(neuron, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分":

            # ==== 組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron // 2)
            self.fc3 = nn.Linear(neuron // 2, neuron // 8)
            self.fc4 = nn.Linear(neuron // 8, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "Multi USER (K>1) - 多層同神經元":

            #==== Multi USER (K>1) - 多層同神經元 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.ln1 = nn.LayerNorm(neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.ln2 = nn.LayerNorm(neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            self.ln3 = nn.LayerNorm(neuron)
            self.fc4 = nn.Linear(neuron, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

        elif network_layer == "Multi USER (K>1) - 多層不同神經元":

            # ==== Multi USER (K>1) - 多層不同神經元 ====
            super(Actor, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.ln1 = nn.LayerNorm(neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.ln2 = nn.LayerNorm(neuron)
            self.fc3 = nn.Linear(neuron, neuron // 2)
            self.ln3 = nn.LayerNorm(neuron // 2)
            self.fc4 = nn.Linear(neuron // 2, neuron // 4)
            self.ln4 = nn.LayerNorm(neuron // 4)
            self.fc5 = nn.Linear(neuron // 4, num_groups * num_phases)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self, state):

        if network_layer == "Single USER (K=1)":

            # ==== Single USER (K=1) ====
            x = torch.relu(self.fc1(state))
            # print(f'x1: {x}')
            x = torch.relu(self.fc2(x))
            # print(f'x2: {x}')
            x = self.fc3(x)
            # print(f'x3: {x}')
            x = x.view(-1, self.num_groups, self.num_phases)  # **確保輸出 shape 為 (batch, num_groups, num_phases)**
            return self.softmax(x)  # **返回每個群組的 phase 機率**

        elif network_layer == "Multi USER (K>1) - 少層同神經元1":

            # ==== Multi USER (K>1) - 少層同神經元1 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "Multi USER (K>1) - 少層同神經元2":

            # ==== Multi USER (K>1) - 少層同神經元2 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = torch.relu(self.fc4(x))
            x = self.fc5(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 1: 少層、同神經元數":

            # ==== 組合 1: 少層、同神經元數 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 2: 多層、同神經元數":

            # ==== 組合 2: 多層、同神經元數 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 3: 多層、不同神經元數":

            # ==== 組合 3: 多層、不同神經元數 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道":

            # ==== 組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block)":

            # ==== 組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block) ====
            x = torch.relu(self.fc1(state))
            residual = x
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x + residual))
            x = self.out(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分":

            # ==== 組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分 ====
            x = torch.relu(self.fc1(state))
            x = torch.relu(self.fc2(x))
            x = torch.relu(self.fc3(x))
            x = self.fc4(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

        elif network_layer == "Multi USER (K>1) - 多層不同神經元" or "Multi USER (K>1) - 多層同神經元":

            # ==== Multi USER (K>1) - 多層不同神經元 or Multi USER (K>1) - 多層同神經元 ====
            x = torch.relu(self.ln1(self.fc1(state)))
            x = torch.relu(self.ln2(self.fc2(x)))
            x = torch.relu(self.ln3(self.fc3(x)))
            x = torch.relu(self.ln4(self.fc4(x)))
            x = self.fc5(x)
            x = x.view(-1, self.num_groups, self.num_phases)
            return self.softmax(x)

class Critic(nn.Module):

    def __init__(self, state_dim, num_groups, num_phases, neuron):

        if network_layer == "Single USER (K=1)":

            # ==== Single USER (K=1) ====
            super(Critic, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            # 狀態網路 (state_dim -> 128)
            self.fc1 = nn.Linear(state_dim, neuron)
            # 動作網路 (num_groups 個離散動作, 每個有 `num_phases` 個選擇)
            self.action_embedding = nn.Embedding(num_phases, 16)  # **將 phase index 轉換為向量**
            self.fc_action = nn.Linear(num_groups * 16, neuron)  # 16 維嵌入, 每個 group 一個
            # 合併狀態與動作特徵
            self.fc2 = nn.Linear(neuron + neuron, neuron)
            self.fc3 = nn.Linear(neuron, 1)  # 輸出 Q 值

        elif network_layer == "Multi USER (K>1) - 少層同神經元1":

            # ==== Multi USER (K>1) - 少層同神經元1 ==== 
            super(Critic, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron * 2, neuron)
            self.fc4 = nn.Linear(neuron, 1)

        elif network_layer == "Multi USER (K>1) - 少層同神經元2":

            # ==== Multi USER (K>1) - 少層同神經元2 ==== 
            super(Critic, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            # 狀態 pathway
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            # 動作 pathway
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron)
            # 合併後 pathway
            self.fc4 = nn.Linear(neuron * 2, neuron)
            self.fc5 = nn.Linear(neuron, 1)

        elif network_layer == "組合 1: 少層、同神經元數":

            # ====  組合 1: 少層、同神經元數 ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron * 2, neuron)
            self.fc4 = nn.Linear(neuron, 1)

        elif network_layer == "組合 2: 多層、同神經元數":

            # ==== 組合 2: 多層、同神經元數 ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron)
            self.fc_merge = nn.Linear(neuron * 2, neuron)
            self.fc_out = nn.Linear(neuron, 1)

        elif network_layer == "組合 3: 多層、不同神經元數":

            # ==== 組合 3: 多層、不同神經元數 ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron // 2)
            self.fc3 = nn.Linear(neuron // 2, neuron // 4)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron // 2)
            self.fc_merge = nn.Linear(neuron // 2 + neuron // 4, neuron // 2)
            self.fc_out = nn.Linear(neuron // 2, 1)

        elif network_layer == "組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道":

            # ==== 組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道 ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron // 2)
            self.fc3 = nn.Linear(neuron // 2, neuron // 4)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.fc_action2 = nn.Linear(neuron, neuron // 2)
            self.fc_merge = nn.Linear((neuron // 4) + (neuron // 2), neuron // 2)
            self.fc_out = nn.Linear(neuron // 2, 1)

        elif network_layer == "組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block)":

            # ==== 組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block) ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc1 = nn.Linear(state_dim, neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.fc3 = nn.Linear(neuron, neuron)
            self.fc_action = nn.Linear(num_groups * 16, neuron)
            self.fc_merge = nn.Linear(2 * neuron, neuron)
            self.out = nn.Linear(neuron, 1)

        elif network_layer == "組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分":

            # ==== 組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分 ====
            super(Critic, self).__init__()
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.shared = nn.Linear(state_dim + num_groups * 16, neuron)
            self.fc1 = nn.Linear(neuron, neuron // 2)
            self.fc2 = nn.Linear(neuron // 2, neuron // 4)
            self.out = nn.Linear(neuron // 4, 1)

        elif network_layer == "Multi USER (K>1) - 多層同神經元":

            # ==== Multi USER (K>1) - 多層同神經元 ====
            super(Critic, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.ln1 = nn.LayerNorm(neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.ln2 = nn.LayerNorm(neuron)
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.ln_action1 = nn.LayerNorm(neuron)
            self.fc_action2 = nn.Linear(neuron, neuron)
            self.ln_action2 = nn.LayerNorm(neuron)
            self.fc3 = nn.Linear(neuron * 2, neuron)
            self.ln3 = nn.LayerNorm(neuron)
            self.fc4 = nn.Linear(neuron, 1)

        elif network_layer == "Multi USER (K>1) - 多層不同神經元":

            # ==== Multi USER (K>1) - 多層不同神經元 ====
            # State pathway
            super(Critic, self).__init__()
            self.num_groups = num_groups
            self.num_phases = num_phases
            self.fc1 = nn.Linear(state_dim, neuron)
            self.ln1 = nn.LayerNorm(neuron)
            self.fc2 = nn.Linear(neuron, neuron)
            self.ln2 = nn.LayerNorm(neuron)
            self.fc3 = nn.Linear(neuron, neuron // 2)
            self.ln3 = nn.LayerNorm(neuron // 2)
            # Action pathway
            self.action_embedding = nn.Embedding(num_phases, 16)
            self.fc_action1 = nn.Linear(num_groups * 16, neuron)
            self.ln_action1 = nn.LayerNorm(neuron)
            self.fc_action2 = nn.Linear(neuron, neuron // 2)
            self.ln_action2 = nn.LayerNorm(neuron // 2)
            # Merge pathway
            self.fc4 = nn.Linear(neuron, neuron // 2)
            self.ln4 = nn.LayerNorm(neuron // 2)
            self.fc5 = nn.Linear(neuron // 2, 1)

    def forward_with_soft_action(self, state, soft_action):
        """
        接收 soft-action (經過 Gumbel-softmax 的 one-hot 向量) 並輸出 Q 值
        soft_action: shape (batch, num_groups, num_phases)
        """

        if network_layer == "Single USER (K=1)":

            # ==== Single USER (K=1) ====
            x_state = torch.relu(self.fc1(state))
            # soft_action: (batch, num_groups, num_phases)
            # 將 one-hot 乘上嵌入表, 取得 weighted embedding
            embed_weights = self.action_embedding.weight  # shape: (num_phases, embed_dim)
            soft_embed = torch.matmul(soft_action, embed_weights)  # (batch, num_groups, embed_dim)
            # 拉平並處理動作特徵
            x_action = soft_embed.view(soft_embed.shape[0], -1)
            x_action = torch.relu(self.fc_action(x_action))
            # 合併後處理
            x = torch.cat([x_state, x_action], dim=-1)
            x = torch.relu(self.fc2(x))
            return self.fc3(x)

        elif network_layer == "Multi USER (K>1) - 少層同神經元1":

            # ==== Multi USER (K>1) - 少層同神經元1 ====
            x_s = torch.relu(self.ln1(self.fc1(state)))
            x_s = torch.relu(self.ln2(self.fc2(x_s)))
            embed_weights = self.action_embedding.weight  # (num_phases, 16)
            soft_embed = torch.matmul(soft_action, embed_weights)  # (batch, num_groups, 16)
            x_a = soft_embed.view(soft_embed.shape[0], -1)          # (batch, num_groups * 16)
            x_a = torch.relu(self.ln_action1(self.fc_action1(x_a)))
            x_a = torch.relu(self.ln_action2(self.fc_action2(x_a)))
            x = torch.cat([x_s, x_a], dim=-1)                       # (batch, neuron * 2)
            x = torch.relu(self.ln3(self.fc3(x)))                   # 正確使用合併後再送入 fc3
            return self.fc4(x) 

        elif network_layer == "Multi USER (K>1) - 少層同神經元2":

            # ==== Multi USER (K>1) - 少層同神經元2 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            embed_weights = self.action_embedding.weight
            soft_embed = torch.matmul(soft_action, embed_weights)
            x_a = soft_embed.view(soft_embed.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc4(x))
            return self.fc5(x)

        elif network_layer == "組合 1: 少層、同神經元數":

            # ==== 組合 1: 少層、同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            embed_weights = self.action_embedding.weight
            x_a = torch.matmul(soft_action, embed_weights).view(soft_action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

        elif network_layer == "組合 2: 多層、同神經元數":

            # ==== 組合 2: 多層、同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            embed_weights = self.action_embedding.weight
            x_a = torch.matmul(soft_action, embed_weights).view(soft_action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 3: 多層、不同神經元數":

            # ==== 組合 3: 多層、不同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            embed_weights = self.action_embedding.weight
            x_a = torch.matmul(soft_action, embed_weights).view(soft_action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道":

            # ==== 組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            embed_weights = self.action_embedding.weight
            soft_embed = torch.matmul(soft_action, embed_weights)
            x_a = torch.relu(self.fc_action1(soft_embed.view(soft_embed.shape[0], -1)))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block)":

            # ==== 組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block) ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            embed_weights = self.action_embedding.weight
            soft_embed = torch.matmul(soft_action, embed_weights)
            x_a = torch.relu(self.fc_action(soft_embed.view(soft_embed.shape[0], -1)))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.out(x)

        elif network_layer == "組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分":

            # ==== 組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分 ====
            embed_weights = self.action_embedding.weight
            soft_embed = torch.matmul(soft_action, embed_weights).view(soft_action.shape[0], -1)
            x = torch.cat([state, soft_embed], dim=-1)
            x = torch.relu(self.shared(x))
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.out(x)

        elif network_layer == "Multi USER (K>1) - 多層不同神經元" or "Multi USER (K>1) - 多層同神經元":

            # ==== Multi USER (K>1) - 多層不同神經元 or Multi USER (K>1) - 多層同神經元 ====
            x_s = torch.relu(self.ln1(self.fc1(state)))
            x_s = torch.relu(self.ln2(self.fc2(x_s)))
            x_s = torch.relu(self.ln3(self.fc3(x_s)))
            embed_weights = self.action_embedding.weight
            soft_embed = torch.matmul(soft_action, embed_weights)
            x_a = soft_embed.view(soft_embed.shape[0], -1)
            x_a = torch.relu(self.ln_action1(self.fc_action1(x_a)))
            x_a = torch.relu(self.ln_action2(self.fc_action2(x_a)))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.ln4(self.fc4(x)))
            return self.fc5(x)

    def forward(self, state, action):
        """
        state: (batch, state_dim)
        action: (batch, num_groups)  # 每個群組選擇的 phase (索引)
        """

        if network_layer == "Single USER (K=1)":

            # ===== Single USER (K=1) =====
            x_state = torch.relu(self.fc1(state))  # 狀態特徵
            # 嵌入 action 並展平
            x_action = self.action_embedding(action)  # shape: (batch, num_groups, 16)
            x_action = x_action.view(x_action.shape[0], -1)  # shape: (batch, num_groups * 16)
            x_action = torch.relu(self.fc_action(x_action))  # 動作特徵
            # 合併狀態與動作
            x = torch.cat([x_state, x_action], dim=-1)
            x = torch.relu(self.fc2(x))
            return self.fc3(x)  # Q 值

        elif network_layer == "Multi USER (K>1) - 少層同神經元1":

            # ==== Multi USER (K>1) - 少層同神經元1  ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_a = self.action_embedding(action)
            x_a = x_a.view(x_a.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

        elif network_layer == "Multi USER (K>1) - 少層同神經元2":

            # ==== Multi USER (K>1) - 少層同神經元2 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            x_a = self.action_embedding(action)
            x_a = x_a.view(x_a.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc4(x))
            return self.fc5(x)

        elif network_layer == "組合 1: 少層、同神經元數":

            # ==== 組合 1: 少層、同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_a = self.action_embedding(action).view(action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc3(x))
            return self.fc4(x)

        elif network_layer == "組合 2: 多層、同神經元數":

            # ==== 組合 2: 多層、同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            x_a = self.action_embedding(action).view(action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 3: 多層、不同神經元數":

            # ==== 組合 3: 多層、不同神經元數 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            x_a = self.action_embedding(action).view(action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道":

            # ==== 組合 4: Actor 使用遞減神經元層數, Critic 使用對應縮減的狀態與動作通道 ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            x_a = self.action_embedding(action).view(action.shape[0], -1)
            x_a = torch.relu(self.fc_action1(x_a))
            x_a = torch.relu(self.fc_action2(x_a))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.fc_out(x)

        elif network_layer == "組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block)":

            # ==== 組合 5: Actor 與 Critic 使用三層殘差連接(Residual Block) ====
            x_s = torch.relu(self.fc1(state))
            x_s = torch.relu(self.fc2(x_s))
            x_s = torch.relu(self.fc3(x_s))
            x_a = torch.relu(self.fc_action(self.action_embedding(action).view(action.shape[0], -1)))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.fc_merge(x))
            return self.out(x)

        elif network_layer == "組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分":

            # ==== 組合 6: Actor 使用瓶頸式壓縮, Critic 使用動作與狀態共享後再拆分 ====
            x_a = self.action_embedding(action).view(action.shape[0], -1)
            x = torch.cat([state, x_a], dim=-1)
            x = torch.relu(self.shared(x))
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.out(x)

        elif network_layer == "Multi USER (K>1) - 多層同神經元":

            # ==== Multi USER (K>1) - 多層同神經元 ====
            x_s = torch.relu(self.ln1(self.fc1(state)))
            x_s = torch.relu(self.ln2(self.fc2(x_s)))
            x_a = self.action_embedding(action)                   # (batch, num_groups, 16)
            x_a = x_a.view(x_a.shape[0], -1)                      # (batch, num_groups * 16)
            x_a = torch.relu(self.ln_action1(self.fc_action1(x_a)))
            x_a = torch.relu(self.ln_action2(self.fc_action2(x_a)))
            x = torch.cat([x_s, x_a], dim=-1)                     # (batch, neuron * 2)
            x = torch.relu(self.ln3(self.fc3(x)))                 # 這裡才用 fc3
            return self.fc4(x)

        elif network_layer == "Multi USER (K>1) - 多層不同神經元":

            # ==== Multi USER (K>1) - 多層不同神經元 ====
            x_s = torch.relu(self.ln1(self.fc1(state)))
            x_s = torch.relu(self.ln2(self.fc2(x_s)))
            x_s = torch.relu(self.ln3(self.fc3(x_s)))
            x_a = self.action_embedding(action)
            x_a = x_a.view(x_a.shape[0], -1)
            x_a = torch.relu(self.ln_action1(self.fc_action1(x_a)))
            x_a = torch.relu(self.ln_action2(self.fc_action2(x_a)))
            x = torch.cat([x_s, x_a], dim=-1)
            x = torch.relu(self.ln4(self.fc4(x)))
            return self.fc5(x) 

class ReplayBuffer:
    def __init__(self, buffer_size, K, state_dim, action_dim, device):
        self.max_size = buffer_size  # 確保 max_size 變數正確命名
        self.ptr = 0
        self.size = 0
        self.device = device

        self.states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        # 這裡的 action_dim 就是 num_groups, 每次的 action 決定的是每個 group 的 phase
        # 所以 action 是一個 shape 為 (num_groups,) 的 array, 
        # 以 1 * 8 為 group 為例, 就會是 [0, 1, 2, 3, 0, 1, 2, 3], 數字代表每個 group 選擇的 phase, 
        # array[0][1][2][3][4][5][6][7] 位置代表 group 1~7
        self.actions = np.zeros((self.max_size, action_dim), dtype=np.float32)      
        self.rewards = np.zeros((self.max_size, K), dtype=np.float32)
        self.next_states = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((self.max_size, 1), dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        index = self.ptr % self.max_size

        self.states[index] = np.array(state, dtype=np.float32)  # 確保 state 為 NumPy float32
        self.actions[index] = np.array(action, dtype=np.int64)  # 確保 action 為 NumPy int64
        self.rewards[index] = reward.cpu().numpy() if isinstance(reward, torch.Tensor) else np.array(reward, dtype=np.float32)
        self.next_states[index] = next_state.cpu().numpy() if isinstance(next_state, torch.Tensor) else np.array(next_state, dtype=np.float32)
        self.dones[index] = np.array(done, dtype=np.bool_)

        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)

        return (
            torch.tensor(self.states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idxs], dtype=torch.int64, device=self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device),
        )

    def __len__(self):
        return self.size

# 將 DDPG 訓練的神經網路儲存
def save_model(checkpoint, dt, args):
    """
    儲存完整的 DDPG 模型, 包括:
    - Actor Network
    - Critic Network
    - Target Networks (Actor + Critic)
    - Optimizers (Actor + Critic)
    - Loss
    """

    if args.multi_seed_run:
        # 確保模型儲存目錄存在
        save_path = './model/seed_sweep/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        # 確保模型儲存目錄存在
        save_path = './model/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    # 確保 `dt` 沒有特殊字元, 例如 ":" (時間格式可能會包含)
    dt = dt.replace(":", "_").replace(" ", "_")

    # 檔案名稱
    model_file = os.path.join(save_path, f"{dt}_seed_{args.seed_rl}_DDPG_model.pth")

    # 儲存 checkpoint
    th.save(checkpoint, model_file)

    print(f"Model saved successfully at: {model_file}")