#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from channel import Channel, Beamformer
import torch.nn.functional as F

from utils import scenario_configs
from typing import Tuple, Union
from utils import gpu, scenario_configs, scenario
from phase_setting import get_phase_config

from math import pi

import pandas as pd
import re  # 用於正則表達式
import time

record_timeslot = 0     # 用來追踪前一個 timeslot 的變數
all_data = []
drawn_scenario = False  # 用於判斷是否已經執行過繪圖
init_csv = False        # 用來追踪是否已經清空過文件
saved_timeslots = {}    # 新增一個字典來追蹤儲存狀態

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


def create_group_mapping(ris_size, group_size):
    """
    根據 RIS 尺寸與群組大小，產生可選擇的群組索引。

    Parameters:
        ris_size (tuple): (rows, cols) 表示 RIS 陣列大小。
        group_size (tuple): (group_rows, group_cols) 表示每個群組的大小。

    Returns:
        num_groups (int): 總共可用的 RIS 群組數量。
        group_mapping (list): 每個群組對應的元素索引。
    """
    rows, cols = ris_size
    group_rows, group_cols = group_size

    num_groups = (rows // group_rows) * (cols // group_cols)

    assert rows % group_rows == 0 and cols % group_cols == 0, "group 形狀必須可以整除 RIS 大小"

    group_mapping = []

    for i in range(0, rows, group_rows):
        for j in range(0, cols, group_cols):
            group = [(i + x) * cols + (j + y) for x in range(group_rows) for y in range(group_cols)]
            group_mapping.append(group)

    return num_groups, group_mapping

def Phase_state(episode, channel, beamformer, folder_name, num_elements, group_mapping, args): 

    global record_timeslot, drawn_scenario

    init_display_settings()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Change working directory to script's location
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    # Draw scenario diagram if not done; mark as completed
    if not drawn_scenario:
        draw_scenario(channel)
        drawn_scenario = True

    # 判斷要 trainnig model 還是 load model, 根據當前 timeslot 是否與前一個不同來決定是否更新通道
    print(f"Timeslot: {episode}\n")
    if record_timeslot==0:  # initial
        init_channel = channel.get_channel_coefficient()
        print(f"RISdata.py/Phase_state || Get \"None\" Channel Coefficient at Timeslot {episode}: {init_channel}[0][:1]")
        channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)          # 這裡的更新channel就是更新3個channel (h_Bk, H_BS, h_Sk), 跟channel有關的東西都會在這裡算出來
        after_update_channel = channel.get_channel_coefficient()          # test: check channel changed
        print(f"RISdata.py/Phase_state || Initial Channel Coefficient[0][:1] at Timeslot {record_timeslot}: {after_update_channel[0][:1]}...\n")
        record_timeslot += 1
    elif record_timeslot == episode:                                                 # initial
        init_channel = channel.get_channel_coefficient()
        print(f"RISdata.py/Phase_state || Get Same Channel Coefficient at Timeslot {episode}: {init_channel[0][:1]}")
        # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)          # 這裡的更新channel就是更新3個channel (h_Bk, H_BS, h_Sk), 跟channel有關的東西都會在這裡算出來
        # after_update_channel = channel.get_channel_coefficient()          # test: check channel changed
        # print(f"RISdata.py/Phase_state || Initial Channel Coefficient[0][:1] at Timeslot {record_timeslot}: {after_update_channel[0][:1]}...\n")
    elif record_timeslot < episode:  
        print(f"Current timeslot: {episode}")
        print(f"Current record_timeslot: {record_timeslot}")
        record_timeslot = episode  # 更新前一個 timeslot 的值
        print(f"Update record_timeslot: {record_timeslot}")

        # channel.update_environment()                                # 這裡的更新環境就是重撒 UE 的位置
        update_env_channel = channel.get_channel_coefficient()
        print(f"RISdata.py/Phase_state || Get Channel Coefficient at Timeslot {episode}: {update_env_channel[0][:1]}")
        # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)
        # after_update_channel = channel.get_channel_coefficient()        # test: check channel changed
        # print(f"RISdata.py/Phase_state || Update UE position[0][:1] at Timeslot {record_timeslot}: {after_update_channel[0][:1]}...\n")    
    else:
        print(f"RISdata.py/Phase_state || Error: record_timeslot {record_timeslot} > Timeslot {episode}.\n")

    Z_switch = torch.ones((1, num_elements), device=args.device)                     # blocking condition, a binary matrix
    # theta_random = torch.rand_like(Z_switch, device=args.device) * 2 * pi            # Generate random values in radians between 0 and 2pi
    # theta_random = torch.zeros_like(Z_switch, device=args.device)                   # DEBUG: phase shift
    # print(f'RISdata.py/Phase_state || Z_switch: {Z_switch}\n')
    # print(f'RISdata.py/Phase_state || theta_random: {theta_random}\n')
    # print(f'RISdata.py/Phase_state || device: {args.device}\n')
    # print(f'RISdata.py/Phase_state || device: {channel.device}\n')

    num_phases, phases_discrete = get_phase_config(args)
    print(f"agent.py/init || num_phases: {num_phases}")
    print(f"agent.py/init || phases_discrete: {phases_discrete}")

    # 初始化 phase
    if args.init_phase_method == 'random':
        # ======== 方式一：Randomly select a group and phase ========
        print("[Phase_state] 初始化模式: random")

        selected_group = group_mapping[np.random.randint(len(group_mapping))]
        phase = np.random.choice(phases_discrete)  # 修正 GPU tensor 轉換問題
        theta_random_radians = np.zeros(num_elements)
        theta_random_radians[selected_group] = phase  # 只影響 selected_group 內的 elements
        print(f"RISdata.py/Phase_state || selected_group: {selected_group}")
        print(f"RISdata.py/Phase_state || phase: {phase}")
        # print(f"RISdata.py/Phase_state || phase_state: {theta_random_radians}\n")

        # num_blocks = num_elements // args.Gelement  # 每 arg.Gelement 個 elements 為 1 組 Group
        # theta_random_indices = torch.randint(0, num_phases, (num_blocks,), device=args.device)  # 產生每個 group 的 phase
        # theta_random_radians = phases_discrete[theta_random_indices]
        # # 將 phase 還原為 64 個 elements
        # theta_random_radians = theta_random_radians.repeat_interleave(args.Gelement)  # 讓每組 phase 連續複製 8 次
        # theta_random_radians = theta_random_radians.unsqueeze(0)  # 變成 (1, 64)
        theta_radians = torch.tensor(theta_random_radians, dtype=torch.float32, device=args.device).unsqueeze(0)  # 變成 (1, 64)
        # theta_random_radians = torch.zeros_like(theta_random).to(args.device)                               # Create a tensor with the same shape as theta_random to store discrete phases
        # print(f'RISdata.py/Phase_state || theta_random_radians: {theta_random_radians}')
        # # Convert continuous phases to discrete phases
        # for j in range(theta_random.size(1)):  # Loop through each RIS elements
        #     distances = torch.abs(theta_random[0, j] - phases_discrete)                     # Calculate the distance to each discrete phase.
        #     min_index = torch.argmin(distances)                                             # Find the index of the nearest discrete phase.
        #     theta_random_radians[0, j] = phases_discrete[min_index]                         # Assign the nearest discrete phase to theta_random_radians
        print(f'RISdata.py/Phase_state || theta_radians BEFORE adjustment: {theta_radians}')
        theta_complex = torch.polar(torch.ones_like(Z_switch, dtype=torch.float32), theta_radians).to(args.device)             # Convert radians to complex numbsers

    elif args.init_phase_method == 'constructive':
        # ======== 方式二：根據 BS→RIS→UE 方向來設定 phase ========
        # 使用幾何方向設定每個 RIS 元素的初始 phase
        # 原理：
        #     目標是讓「從 BS 發出的電波 → 反射到每個 RIS → 傳到 UE」的路徑上，
        #     每條波都剛好同步到達 UE，讓接收到的訊號加強（constructive interference）
        #
        # 做法：
        #     每個 RIS element 計算兩段距離：
        #         - 從 BS 到這個 RIS 的距離
        #         - 從這個 RIS 到 UE 的距離
        #     這兩段相加，代表整條波路的長度
        #     把這段距離除以波長，就能換算成要補償的 phase（弧度）
        #     最後乘上 -2π，是因為我們要「反向補償」那段 propagation delay
        #
        # 白話文：
        #     對每個 RIS element 計算從 BS → RIS element → UE 的總距離
        #     為了讓所有路徑的反射波在 UE 處同步到達（即相位一致），
        #     我們需要補償這段距離造成的 phase 差。
        #     因此，對每個 element 設定的 phase 為：
        #         phi_n = - 2π / λ * (d_BS_to_RIS_n + d_RIS_n_to_UE)
        #     這個 phase 補償會讓所有波在 UE 處達到 constructive interference，
        #     增強接收效果，就像主動對準 UE 的波束一樣（beamforming）。
        #
        # 公式：
        #     phi_n = -2 * pi * (|r_n - BS_pos| + |r_n - UE_pos|) / wavelength

        print("[Phase_state] 初始化模式: beam (directional beam steering)")
        
        # # 取得 Base Station、RIS 各元素位置、User Equipment 的位置
        # BS_pos = channel.BS_pos.reshape(1, 3).to(args.device)              # BS 位置 (1, 3)
        # RIS_pos = channel.ris_pos.reshape(num_elements, 3).to(args.device)  # RIS 所有元素位置 (N, 3)
        # UE_pos = channel.MU_pos.to(args.device)                             # UE 位置 (K, 2) or (K, 3)

        # # 若 UE 位置只有 x, y，補上 z=0，成為 (K, 3)
        # if UE_pos.shape[1] == 2:
        #     ue_z = torch.zeros((UE_pos.shape[0], 1), device=args.device)
        #     UE_pos = torch.cat([UE_pos, ue_z], dim=1)

        # # 以所有 UE 的重心作為目標反射方向
        # UE_center = torch.mean(UE_pos, dim=0, keepdim=True)  # (1, 3)

        # # 入射方向向量：BS 到 RIS 每個元素
        # v_incident = RIS_pos - BS_pos  # (N, 3)
        # # 反射方向向量：RIS 每個元素到 UE 重心
        # v_reflect = UE_center - RIS_pos  # (N, 3)

        # # 取得 RIS 法向量（假設所有 RIS 元素共用同一方向）
        # n_hat = channel.ris_norm.reshape(1, 3).to(args.device)
        # n_hat = n_hat / torch.norm(n_hat)  # 單位化

        # # 單位向量化函數
        # def unit(x): return x / torch.norm(x, dim=1, keepdim=True)

        # # 單位入射向量與反射向量
        # u_incident = unit(v_incident)
        # u_reflect = unit(v_reflect)

        # # 期望的反射方向：以 n_hat 為鏡面反射法則，計算預期出射向量
        # u_reflect_dir = u_incident - 2 * (torch.sum(u_incident * n_hat, dim=1, keepdim=True)) * n_hat  # (N, 3)

        # # 計算實際方向與理想方向之間的夾角，反映 phase shift 需求
        # dot = torch.sum(unit(u_reflect) * unit(u_reflect_dir), dim=1)  # 內積計算夾角餘弦值
        # dot = torch.clamp(dot, -1.0, 1.0)  # 防止數值超出範圍
        # theta = torch.acos(dot)  # 反餘弦取得相位角度 (弧度)，(N,)

        # # 定義離散相位集合，這裡為 [0, π]
        # phases_discrete = torch.tensor([0, np.pi], device=args.device)

        # # 每個元素挑選距離最近的離散相位
        # theta_discrete = torch.argmin(torch.abs(theta.view(-1, 1) - phases_discrete), dim=1)  # (N,)
        # theta_final = phases_discrete[theta_discrete]  # (N,)
        # torch.set_printoptions(threshold=float('inf'))
        # print(f'RISdata.py/Phase_state || theta_final: {theta_final}')
        # # 最終相位轉為極座標表示，模為1、角度為離散後相位
        # theta_complex = torch.polar(torch.ones_like(theta_final), theta_final).reshape(1, num_elements)
        
        # Step 1: Get positions
        BS_pos = channel.BS_pos.reshape(1, 3).to(args.device)
        RIS_pos = channel.ris_pos.reshape(num_elements, 3).to(args.device)
        UE_pos = channel.MU_pos.to(args.device)
        if UE_pos.shape[1] == 2:
            ue_z = torch.zeros((UE_pos.shape[0], 1), device=args.device)
            UE_pos = torch.cat([UE_pos, ue_z], dim=1)

        UE_center = torch.mean(UE_pos, dim=0, keepdim=True)
        print(f'BS_pos: {BS_pos}')
        print(f'RIS_pos (first 5): {RIS_pos[:5]}')
        print(f'UE_center: {UE_center}')

        # Step 2: Compute vectors
        v_in = F.normalize(RIS_pos - BS_pos, dim=1)        # BS → RIS
        v_out = F.normalize(UE_center - RIS_pos, dim=1)    # RIS → UE center
        v_total = v_in + v_out
        print(f'[Phase_state] v_in: {v_in}')
        print(f'[Phase_state] v_out: {v_out}')
        print(f'[Phase_state] v_total: {v_total}')

        # Step 3: Phase shift from projection
        path_diff = (RIS_pos * v_total).sum(dim=1)         # 投影到方向向量上
        wavelength = channel.wavelength
        theta = -2 * np.pi * path_diff / wavelength

        # Step 4: Discretize phase
        if isinstance(phases_discrete, np.ndarray):
            phases_discrete = torch.tensor(phases_discrete, dtype=torch.float32, device=args.device)
        else:
            phases_discrete = phases_discrete.to(args.device)
        theta_discrete_idx = torch.argmin(torch.abs(theta.view(-1, 1) - phases_discrete), dim=1)
        theta_final = phases_discrete[theta_discrete_idx]
        torch.set_printoptions(threshold=float('inf'))
        print(f'[Phase_state] theta_final: {theta_final}')

        # Step 5: Convert to complex
        theta_complex = torch.polar(torch.ones_like(theta_final), theta_final).reshape(1, num_elements)
        # print(f'[Phase_state] theta_complex: {theta_complex}')
        exit()
    else:
        raise ValueError(f"[Phase_state] Unsupported phase_init_mode: {args.phase_init_mode}")

    Z_theta = Z_switch * theta_complex
    # print(f'RISdata.py/Phase_state || Z_theta: {Z_theta}\n')

    B_BS, B_Sk, B_Bk = channel.getBlockMats()  # Blocking condition
    D_BS, D_Sk, D_Bk = channel.getDistMats()  # Distances (m)

    H_random = channel.get_joint_channel_coefficient(Z_theta) 
    W_random = beamformer.MRT(H_random)

    sinr_linear = channel.SINR(H_random, W_random)
    sinr_db = 10 * torch.log10(sinr_linear).cpu().numpy()
    # print(f'RISdata.py/Phase_state || SINR(linear) BEFORE adjustment: {sinr_linear}')
    # print(f'RISdata.py/Phase_state || SINR(dB) BEFORE adjustment: {sinr_db_random}')

    # Ensure all are tensors, convert SINR to a tensor
    sinr_db_tensor = torch.from_numpy(sinr_db).to(args.device)
    print(f'RISdata.py/Phase_state || SINR(dB) BEFORE adjustment: {sinr_db_tensor}')

    # Concatenate phase and SINR along dim=1.
    Phase_state = torch.cat((theta_radians, sinr_db_tensor), dim=1).squeeze()
    # print(f'RISdata.py/Phase_state || Phase_state: {Phase_state[:]} \nshape: {Phase_state.shape[:]}...\n')       # ([1, 1025])

    # Save the Phase and UE of each timeslot to a CSV
    timestamp_folder = os.path.basename(folder_name)
    save_timeslot_phase(episode, Phase_state, scenario, num_elements, timestamp_folder)

    # Datarate
    if sinr_linear is not None:
        # Calculate channel capacity based on Shannon's theorem
        # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
        channel_capacity = torch.log2(1 + sinr_linear)
        # Assume each symbol can carry 1 bit
        # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
        datarate = channel_capacity
    else:
        datarate = 0
    print(f"RISdata.py//Phase_state || Datarate BEFORE adjustment: {datarate}\n")

    return(Phase_state, sinr_linear, sinr_db, datarate)

def save_timeslot_phase(episode, Phase_state, scenario, num_elements, timestamp_folder):

    global all_data, init_csv, saved_timeslots     # Store all timeslot data for single CSV export at the end

    init_display_settings()

    # If it is timeslot1, do not save data
    if episode == 0 or episode in saved_timeslots:
        # print(f"Timeslot {timeslot} 已經儲存過, 跳過存檔...")
        return  # 跳過儲存, 避免重複

    # 將當前 timeslot 標記為已儲存
    saved_timeslots[episode] = True

    Phase_state_np = Phase_state.cpu().numpy()  # Convert tensor to NumPy and move to CPU
    # print(f"Phase_state_np.shape: {Phase_state_np.shape}\n")

    # Save CSV path: Ensure the directory exists
    random_csv_path = f'./generate_RIS/timeslot_phase/{scenario}/{timestamp_folder}'
    os.makedirs(random_csv_path, exist_ok=True)

    # Clear existing CSV files on first execution
    if not init_csv:
        clear_existing_csv(random_csv_path, num_elements, episode)
        init_csv = True

    # --- Save each timeslot individually with and without headers ---
    row_data = Phase_state_np
    # print(f"Phase_state_np[0]: {Phase_state_np[0]}\n")

    num_ues = Phase_state_np.shape[0] - num_elements
    column_names = [f'Element{j+1}' for j in range(num_elements)] + \
                [f'UE{k+1}\'s_SINR' for k in range(num_ues)]

    # Save without headers
    random_df = pd.DataFrame([row_data])
    eachtime_csv_path = os.path.join(random_csv_path, f'RIS_element_{num_elements}_Phase_timeslot{episode}.csv')
    random_df.to_csv(eachtime_csv_path, index=False, header=False)
    # print(f'Saved timeslot {i+1} to {eachtime_csv_path}.')

    # Save with headers
    random_df_marked = pd.DataFrame([row_data], columns=column_names)
    eachtime_csv_path_marked = os.path.join(random_csv_path, f'RIS_element_{num_elements}_Phase_timeslot{episode}_marked.csv')
    random_df_marked.to_csv(eachtime_csv_path_marked, index=True, header=True)
    # print(f'Saved timeslot {i+1} to {eachtime_csv_path_marked}.')

    # --- Save all accumulated timeslot data to a single CSV (with and without headers) ---
    # Collect data for saving all timeslots to a single file later
    all_data.append(row_data)

    index_names = [f'timeslot_{m+1}' for m in range(len(all_data))]
    
    # Save without headers
    alltime_df_all = pd.DataFrame(all_data)
    alltime_csv_path = os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots.csv')
    alltime_df_all.to_csv(alltime_csv_path, index=False, header=False)

    # Save with headers
    alltime_df_all_marked = pd.DataFrame(all_data, columns=column_names, index=index_names)
    alltime_csv_path_marked = os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots_marked.csv')
    alltime_df_all_marked.to_csv(alltime_csv_path_marked, index=True, header=True)

    # print(f'RIS_element_Phase for all timeslots saved to {eachtime_csv_path} and {alltime_csv_path}\n')


def draw_scenario(channel):
    # print('=====================Scenario diagram drawn and saved=====================')

    init_display_settings()

    # Ensure the result directory exists; if not, create it
    dir = os.path.abspath(f'./Scenario/{scenario}')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # Plot and save the blockage conditions and overview of the scenario
    channel.plot_block_cond(dir=dir)  # Plot block condition
    channel.show(dir=dir)  # plot the scenario

    # print(f"Scenario diagram drawn and saved.\n")

def clear_existing_csv(random_csv_path, num_elements, timeslot):
    
    init_display_settings()

    # Cleared CSV list
    csv_to_clear = [
        os.path.join(random_csv_path, f'RIS_element_{num_elements}_Phase_timeslot{timeslot}.csv'),
        os.path.join(random_csv_path, f'RIS_element_{num_elements}_Phase_timeslot{timeslot}_marked.csv'),
        os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots.csv'),
        os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots_marked.csv')
    ]

    # Cleared CSV
    for clear_path in csv_to_clear:
        if os.path.exists(clear_path):
            os.remove(clear_path)
            # print(f"Cleared existing file: {clear_path}")

    # print("\n")

# 測試 RISstate 方法, 印出結果
if __name__ == '__main__':
    a, sinr_linear, sinr_db = Phase_state(1)                                                               # for test
    # print(a)


'''''''''''''''''''''''''''''''''''''''''''''''''''
RISstate data:
    Theta 1, Theta 2, ... , Theta 1024, SINR

每個 element 都會有自己的 1 個 Theta (= Phase), 所以 element 數量 = Theta 數量 = 32*32 = 1024
'''''''''''''''''''''''''''''''''''''''''''''''''''