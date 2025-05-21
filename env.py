#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from RISdata import Phase_state

import torch
from channel import Channel, Beamformer
from utils import scenario_configs
from utils import gpu, scenario
from math import pi

import pandas as pd

def init_display_settings():
    np.set_printoptions(
        precision=8,                     # 控制小數點顯示的位數
        suppress=True,                   # 不使用科學計數法
        floatmode='fixed',               # 顯示固定小數位數
        threshold=np.inf,                # 陣列中元素的顯示數量
        edgeitems=30,                    # 當陣列太大需要省略顯示時，控制在開始和結尾處顯示的元素數量
        linewidth = 1000000              # 控制輸出時每行的最大字元數，避免換行
    )
    torch.set_printoptions(precision=8)
    pd.set_option('display.precision', 8)
    pd.set_option('display.float_format', lambda x: '%.8f' % x)

class RISEnv:

    def __init__(self, episode, channel, beamformer, num_elements, num_groups, num_phases, group_mapping, phases_discrete, args):                               # 初始化環境
        init_display_settings()

        self.device = args.device
        self.episode = episode
        self.channel = channel
        self.beamformer = beamformer
        self.num_elements = num_elements
        self.num_groups = num_groups
        self.num_phases = num_phases
        self.group_mapping = group_mapping
        self.phases_discrete = phases_discrete
        self.Phase_state = 0

        # print(f'env.py/__init__ || self.device: {self.device}')
        # print(f'env.py/__init__ || self.episode: {self.episode}')
        # print(f'env.py/__init__ || self.channel: {self.channel}')
        # print(f'env.py/__init__ || self.beamformer: {self.beamformer}')
        # print(f'env.py/__init__ || self.num_elements: {self.num_elements}')
        # print(f'env.py/__init__ || self.num_groups: {self.num_groups}')
        # print(f'env.py/__init__ || self.num_phases: {self.num_phases}')
        # print(f'env.py/__init__ || self.Phase_state: {self.Phase_state}')

    def reset(self, episode, agent, num_elements, channel, beamformer, group_mapping, folder_name, args, record_timeslot):                                        # 初始化環境狀態條件
        init_display_settings()

        # DEBUG
        print(f'env.py/reset || channel: {channel.get_channel_coefficient()}')
        print(f"env.py/reset || record_timeslot: {record_timeslot['val']}")

        state_before, sinr_before_linear, sinr_before_db, datarate_before = Phase_state(episode, channel, beamformer, folder_name, num_elements, group_mapping, args, record_timeslot)                             # 獲取這個timeslot的初始phase
        # print(f"env.py/reset || state_before: {state_before}")
        # print(f"env.py/reset || sinr_linear_random: {sinr_before_linear}")
        # print(f"env.py/reset || sinr_db_random: {sinr_before_db}")
        # print(f"env.py/reset || datarate_before: {datarate_before}\n")

        # print(f"self.Phase_state: {self.Phase_state}")
        self.Phase_state = state_before                                       # 更新狀態        
        # print(f"self.Phase_state: {self.Phase_state}")
        state_before = state_before.tolist()                                    # 轉換為 list
        return state_before, sinr_before_linear, sinr_before_db, datarate_before                                                   # 返回當前的狀態

    def step(self, actions, agent, num_elements):                                               # 根據動作計算下一狀態和獎勵
        init_display_settings()

        # print(f"Element(Agent): {agent+1}")                                       # index從0開始算, 所以print時+1較符合人類直覺
        reward = []                                                           # 初始化獎勵
        done = []                                                             # 初始化完成標誌

        print(f"env.py/step || Get RIS element BEFORE adjustment: {self.Phase_state[:]}")
        # print(f"ent.py/step || actions: {actions}")
        # print(f'env.py/step || self.group_mapping: {self.group_mapping}, type: {type(self.group_mapping)}')

        # `actions` 現在是 num_groups 個 phase choices (ex: [3, 0, 0, 0, 2, 3, 0, 3])
        action_phase = np.zeros_like(self.Phase_state[:num_elements].cpu().numpy())
        # print(f"env.py/step || action_phase: {action_phase}")

        for group_idx, phase_idx in enumerate(actions):
            if group_idx >= len(self.group_mapping):
                print(f"Warning: group_idx {group_idx} 超出 group_mapping 長度 {len(self.group_mapping)}")
                continue  # 跳過無效 group_idx

            selected_group = self.group_mapping[group_idx]  # 取得對應的 group 內的 elements
            if not isinstance(selected_group, (list, np.ndarray)):
                raise TypeError(f"selected_group 應該是 list, 但得到 {type(selected_group)}: {selected_group}")

            if phase_idx < 0 or phase_idx >= len(self.phases_discrete):  
                raise ValueError(f"phase_idx {phase_idx} 超出 phases_discrete 範圍 {len(self.phases_discrete)}")

            phase_value = self.phases_discrete[phase_idx]  # 正確映射 phase 值
            action_phase[selected_group] = phase_value     # 更新 group 內所有 elements 的 phase
        # print(f"env.py/step || action_phase: {action_phase}")

        # 將 NumPy 轉回 PyTorch tensor 並保持在 CUDA
        self.Phase_state[:num_elements] = torch.tensor(action_phase, dtype=torch.float32, device=self.device)
        # print(f"env.py/step || Updated Phase_state: {self.Phase_state[:-1]}")

        # 執行動作, 更新狀態
        # self.Phase_state[:num_elements] = torch.tensor(actions, dtype=torch.float32, device=self.device)
        # print(f"env.py/step || RIS element [:{num_elements}] AFTER adjustment: {self.Phase_state[:num_elements]}...\n")

        # 獲取通道係數
        get_channel = self.channel.get_channel_coefficient()
        print(f"env.py/step || Get Channel Coefficient[0][:1]: {get_channel[0][:1]}")

        Z = torch.ones((1, num_elements), device=self.device)                               # blocking condition, a binary matrix
        # print(f"env.py/step || Z: {Z}")
        Theta_adj_re_tensor = self.Phase_state[:num_elements].clone().detach().unsqueeze(0)
        # Theta_adj_re_tensor = torch.tensor(self.Phase_state[:-1], dtype=torch.float32, device=self.device).unsqueeze(0)                                            # 將 NumPy array 轉換為 list 並封裝在另一個 list 中
        # Theta_adj_re_tensor = torch.tensor(actions, dtype=torch.float32, device=self.device).unsqueeze(0)                                            # 將 NumPy array 轉換為 list 並封裝在另一個 list 中
        # print(f"env.py/step || Theta_adj_re_tensor: {Theta_adj_re_tensor}")

        Theta_adj_re_complex_Phase = torch.polar(torch.ones_like(Z, dtype=torch.float32), Theta_adj_re_tensor).to(self.device)          # 生成極座標形式(複數)
        # print(f"env.py/step || Theta_adj_re_complex_Phase: {Theta_adj_re_complex_Phase}")

        Z_Theta = Z * Theta_adj_re_complex_Phase
        # print(f"env.py/step || Z_Theta: {Z_Theta}")

        H = self.channel.get_joint_channel_coefficient(Z_Theta)
        W = self.beamformer.MRT(H)

        # 計算SINR和Datarate
        sinr = self.channel.SINR(H, W)                                                      # linear
        # print(f"env.py/step || SINR(linear): {sinr}, shape: {sinr.shape}")
        sinr_db = 10 * torch.log10(sinr).cpu().numpy().reshape(-1) 
        # print(f"env.py/step || SINR(dB): {sinr_db}, SINR(linear): {sinr}, shape: {sinr_db.shape}")

        # 更新state中最後一個值: SINR
        self.Phase_state[num_elements:] = torch.tensor(sinr_db, device=self.Phase_state.device, dtype=torch.float64)
        print(f"env.py/step || RIS element AFTER adjustment: {self.Phase_state}")

        if sinr is not None:
            # Calculate channel capacity based on Shannon's theorem
            # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
            channel_capacity = torch.log2(1 + sinr)
            # print(f"env.py/step || Channel capacity: {channel_capacity}, shape: {channel_capacity.shape}")
            # Assume each symbol can carry 1 bit
            # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
            datarate = channel_capacity.squeeze(1) 
        else:
            datarate = 0

        # print(f"env.py/step || Datarate AFTER adjustment: {datarate}")
        # reward
        reward = datarate
        # print(f"env.py/step || Reward: {reward}, type: {type(reward)}")

        # # 放大 reward, 使其為非線性
        # reward_exp = np.exp(self.alpha * reward.cpu().numpy()) - 1
        # # print(f"env.py/step || Reward_exp: {reward_exp}, type: {type(reward_exp)}")
        # reward_E = torch.FloatTensor(reward_exp).unsqueeze(1).to(self.device)
        # print(f"env.py/step || Reward_exp: {reward_E}, type: {type(reward_E)}")

        next_state = self.Phase_state
        # print(f"next_state: {next_state}")
        # print(f"next_state type: {type(next_state[0])}")
        # done.append(True)  # 在這個場景中, 每步都可能是結束, 因為每個timeslot被當作獨立的episode

        curr_state = self.Phase_state
        # print(f"env.py/step || curr_state: {curr_state[:]}...\n")

        done.append(False) 

        return next_state, reward, done, sinr, sinr_db, curr_state                 # 返回更新的狀態、獎勵、完成狀態
        # return next_state, reward_E, done, sinr, sinr_db, curr_state                 # 返回更新的狀態、獎勵、完成狀態
