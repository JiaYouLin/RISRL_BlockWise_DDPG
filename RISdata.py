#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from channel import Channel, Beamformer
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

    Z_random = torch.ones((1, num_elements), device=args.device)                     # blocking condition, a binary matrix
    # theta_random = torch.rand_like(Z_random, device=args.device) * 2 * pi            # Generate random values in radians between 0 and 2pi
    # theta_random = torch.zeros_like(Z_random, device=args.device)                   # DEBUG: phase shift
    # print(f'RISdata.py/Phase_state || Z_random: {Z_random}\n')
    # print(f'RISdata.py/Phase_state || theta_random: {theta_random}\n')
    # print(f'RISdata.py/Phase_state || device: {args.device}\n')
    # print(f'RISdata.py/Phase_state || device: {channel.device}\n')

    num_phases, phases_discrete = get_phase_config(args)
    print(f"agent.py/init || num_phases: {num_phases}")
    print(f"agent.py/init || phases_discrete: {phases_discrete}")

    # Randomly select a group and phase
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
    theta_random_radians = torch.tensor(theta_random_radians, dtype=torch.float32, device=args.device).unsqueeze(0)  # 變成 (1, 64)
    # theta_random_radians = torch.zeros_like(theta_random).to(args.device)                               # Create a tensor with the same shape as theta_random to store discrete phases
    # print(f'RISdata.py/Phase_state || theta_random_radians: {theta_random_radians}')
    # # Convert continuous phases to discrete phases
    # for j in range(theta_random.size(1)):  # Loop through each RIS elements
    #     distances = torch.abs(theta_random[0, j] - phases_discrete)                     # Calculate the distance to each discrete phase.
    #     min_index = torch.argmin(distances)                                             # Find the index of the nearest discrete phase.
    #     theta_random_radians[0, j] = phases_discrete[min_index]                         # Assign the nearest discrete phase to theta_random_radians
    print(f'RISdata.py/Phase_state || theta_random_radians BEFORE adjustment: {theta_random_radians}')

    theta_random_complex = torch.polar(torch.ones_like(Z_random, dtype=torch.float32), theta_random_radians).to(args.device)             # Convert radians to complex numbsers
    Z_theta_random = Z_random * theta_random_complex
    # print(f'RISdata.py/Phase_state || Z_theta_random: {Z_theta_random}\n')

    B_BS, B_Sk, B_Bk = channel.getBlockMats()  # Blocking condition
    D_BS, D_Sk, D_Bk = channel.getDistMats()  # Distances (m)

    H_random = channel.get_joint_channel_coefficient(Z_theta_random) 
    W_random = beamformer.MRT(H_random)

    sinr_linear_random = channel.SINR(H_random, W_random)
    sinr_db_random = 10 * torch.log10(sinr_linear_random).cpu().numpy()
    # print(f'RISdata.py/Phase_state || SINR(linear) BEFORE adjustment: {sinr_linear_random}')
    # print(f'RISdata.py/Phase_state || SINR(dB) BEFORE adjustment: {sinr_db_random}')

    # Ensure all are tensors, convert SINR to a tensor
    sinr_db_random_tensor = torch.from_numpy(sinr_db_random).to(args.device)
    print(f'RISdata.py/Phase_state || SINR(dB) BEFORE adjustment: {sinr_db_random_tensor}')

    # Concatenate phase and SINR along dim=1.
    Phase_state = torch.cat((theta_random_radians, sinr_db_random_tensor), dim=1).squeeze()
    # print(f'RISdata.py/Phase_state || Phase_state: {Phase_state[:]} \nshape: {Phase_state.shape[:]}...\n')       # ([1, 1025])

    # Save the Phase and UE of each timeslot to a CSV
    timestamp_folder = os.path.basename(folder_name)
    save_timeslot_phase(episode, Phase_state, scenario, num_elements, timestamp_folder)

    # Datarate
    if sinr_linear_random is not None:
        # Calculate channel capacity based on Shannon's theorem
        # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
        channel_capacity = torch.log2(1 + sinr_linear_random)
        # Assume each symbol can carry 1 bit
        # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
        datarate = channel_capacity
    else:
        datarate = 0
    print(f"RISdata.py//Phase_state || Datarate BEFORE adjustment: {datarate}\n")

    return(Phase_state, sinr_linear_random, sinr_db_random, datarate)

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