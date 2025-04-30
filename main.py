#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尚桓共有27個agent (leader), 1個agent要做5個動作(5種訊息), 這5個動作裡面有16種方式(CQI)可以做
我:
    single agent: ris面板作為1個agent, 1個agent下有32*32個動作(element), 這32*32個動作中有4種角度(1個不做等於維持原角度)
    multi agent: 
        (1) element作為32*32個agent, 每個agent下只有1個動作(Theta(Phase)做調整), 這1個動作中有4種角度(1個不做等於維持原角度)
        (2) element作為32*32個agent, 每個agent下只有2個動作(Theta(Phase)做調整, Z開或關), 這1個動作中有4種角度(1個不做等於維持原角度)
"""

import argparse, datetime, time
import warnings
import os
import gc
import json
from itertools import product
import math
from math import pi
import random
import numpy as np
import pandas as pd
import torch

from agent import DDPGAgent
from RISdata import Phase_state, create_group_mapping
from env import RISEnv
from utils import scenario_configs, scenario
from channel import Channel, Beamformer
from output import SaveCSV, SaveCSV_testing
import network
from phase_setting import get_phase_config

# channel_beam function
import itertools    
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

warnings.filterwarnings("ignore", category=FutureWarning)

def init_display_settings():
    np.set_printoptions(
        precision=8,                     # 控制小數點顯示的位數
        suppress=True,                   # 不使用科學計數法
        floatmode='fixed',               # 顯示固定小數位數
        threshold=np.inf,                # 陣列中元素的顯示數量
        edgeitems=30,                    # 當陣列太大需要省略顯示時, 控制在開始和結尾處顯示的元素數量
        linewidth = 1000000              # 控制輸出時每行的最大字元數, 避免換行
    )
    torch.set_printoptions(precision=8)
    pd.set_option('display.precision', 8)
    pd.set_option('display.float_format', lambda x: '%.8f' % x)

# def generate_all_combinations(a_dim, num_elements, total_combinations):
#     """
#     使用進制轉換生成所有組合。
    
#     Args:
#         a_dim: element數量, 會作為組合數量的基數, 如 4^4 次方
#         num_elements: RIS element 數量
#         total_combinations: 要生成的組合總數
    
#     Returns:
#         一個包含所有組合的生成器, 每個組合都是長度為 num_elements 的 list
#     """
#     for index in range(total_combinations):
#         combination = []
#         temp_index = index
#         for _ in range(num_elements):
#             combination.append(temp_index % a_dim)  # 計算當前位的值
#             temp_index //= a_dim                    # 更新索引
#         yield list(reversed(combination))          # 反轉組合順序以符合進制轉換法

# def noris(channel, beamformer, scenario, num_elements, K, args.device):
    
#     init_display_settings()
    
#     print("==================Test 0 (NO RIS, Z=0)==================")    

#     Z_noris = torch.zeros((1, num_elements)).type(torch.float32).to(args.device)
#     # print(f'Z_noris: {Z_noris} \nshape: {Z_noris.shape}\n')

#     Z_noris_complex = torch.polar(Z_noris, torch.zeros_like(Z_noris)).to(args.device)
#     # print(f'Z_noris_complex: {Z_noris_complex} \nshape: {Z_noris_complex.shape}\n')

#     Z_theta_noris = Z_noris * Z_noris_complex
#     # print(f'Z_theta_noris: {Z_noris_complex} \nshape: {Z_theta_noris.shape}\n')

#     # B_BS, B_Sk, B_Bk = channel.getBlockMats()
#     # D_BS, D_Sk, D_Bk = channel.getDistMats()

#     # H_noris, AWGN_norm_noris = channel.coef(Z_theta_noris, batch_size=None, progress=True)
#     H_noris = channel.get_joint_channel_coefficient(Z_theta_noris)
#     W_noris = beamformer.MRT(H_noris)
#     # print(f'H_noris: {H_noris} \nshape: {H_noris.shape}\n')

#     # print(f'B_BS: {B_BS}, B_BS shape: {B_BS.shape}')      # DEBUG: BS-RIS的阻擋矩陣
#     # print(f'B_Bk: {B_Bk}, B_Bk shape: {B_Bk.shape}')      # DEBUG: BS-UE的阻擋矩陣
#     # print(f'B_Sk: {B_Sk}, B_Sk shape: {B_Sk.shape}')      # DEBUG: RIS-UE的阻擋矩陣
#     # print(f'D_BS shape: {D_BS.shape}, D_Sk shape: {D_Sk.shape}, D_Bk shape: {D_Bk.shape}')
#     # print(f'H shape: {H_noris.shape}, type: {type(H_noris)}')
#     # print(f'W shape: {W_noris.shape}, type: {type(W_noris)}')

#     # 計算SINR和Datarate
#     sinr_linear_noris = channel.SINR(H_noris, W_noris)
#     sinr_db_noris = 10*torch.log10(sinr_linear_noris).cpu().numpy()  
#     print(f"No RIS SINR: {sinr_db_noris}, shape: {sinr_db_noris.shape}, type: {type(sinr_db_noris)} \n")

#     if sinr_linear_noris is not None:
#         # Calculate channel capacity based on Shannon's theorem
#         # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
#         channel_capacity_noris = torch.log2(1 + sinr_linear_noris)
#         # Assume each symbol can carry 1 bit
#         # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
#         datarate_noris = channel_capacity_noris
#     else:
#         datarate_noris = 0
#     print(f"No RIS Datarate: {datarate_noris}, shape: {datarate_noris.shape}, type: {type(datarate_noris)} \n")

#     # Calculate the Avg. SINR, Total Datarate, and Avg. Datarate for UE 1~16.
#     avg_sinr_noris = np.mean(sinr_db_noris[:K])
#     avg_datarate_noris = datarate_noris[:K].mean().cpu().item()
#     total_datarate_noris = datarate_noris[:K].sum().cpu().item()

#     # Save each UE's SINR and Datarate to the dataframe
#     df_noris = pd.DataFrame()
#     for i in range(len(sinr_db_noris[0])):
#         df_noris[f'UE{i+1}\'s_SINR'] = [sinr_db_noris[0][i]]
#     for i in range(len(datarate_noris[0])):
#         df_noris[f'UE{i+1}\'s_Datarate'] = [datarate_noris[0][i].cpu().numpy()]
#     df_noris['Average_SINR'] = [avg_sinr_noris]
#     df_noris['Average_Datarate'] = [avg_datarate_noris]
#     df_noris['Total_Datarate'] = [total_datarate_noris]

#     # Save to CSV
#     csv_path_noris = os.path.abspath(f'./Scenario/{scenario}')
#     if not os.path.isdir(csv_path_noris):
#         os.makedirs(csv_path_noris)
#     file_path_noris = os.path.join(csv_path_noris, 'noris_sinr_datarate.csv')
#     df_noris.to_csv(file_path_noris, index=False)           # index=False means not writing the row index

#     print(f'No RIS: SINR and Datarate without RIS saved to {csv_path_noris}.')

def block_wise_phase_grouping(channel, beamformer, scenario, folder_name, num_elements, K, device, count_tmp, args):
    '''
    將RIS元素分成G個群組, 每組共享相同的phase, 遍歷所有組合, 計算每種組合的SINR和Datarate
    遍歷所有 phase 組合, 逐一計算並即時寫入 CSV
    '''

    init_display_settings()

    num_phases, phases_discrete = get_phase_config(args)
    print(f"agent.py/init_agent || num_phases: {num_phases}")
    print(f"agent.py/init_agent || phases_discrete: {phases_discrete}")

    print("================== Test 7 (Block-wise Phase Grouping) ===================")    

    # 解析 `group_size` 設定, 確保格式正確
    if not isinstance(args.group_size, tuple) or len(args.group_size) != 2:
        raise ValueError("args.group_size 應該是 tuple (row_group, col_group), 例如 (1, 8) 或 (4, 4)")

    group_rows, group_cols = args.group_size  # 群組的行數與列數
    ris_rows, ris_cols = int(np.sqrt(num_elements)), int(np.sqrt(num_elements))  # RIS 的行列數（假設 8x8）

    if ris_rows % group_rows != 0 or ris_cols % group_cols != 0:
        raise ValueError(f"群組大小 {args.group_size} 不能均分 RIS 陣列 {ris_rows}x{ris_cols}")

    num_blocks = (ris_rows // group_rows) * (ris_cols // group_cols)  # 總群組數
    elements_per_block = group_rows * group_cols  # 每個群組的 RIS 元素數量

    print(f"----- 測試 num_blocks = {num_blocks}, elements_per_block = {elements_per_block} -----")

    # 建立正確的 Group Mapping
    group_mapping = []
    for i in range(0, ris_rows, group_rows):  # row-wise
        for j in range(0, ris_cols, group_cols):  # col-wise
            block = []
            for x in range(group_rows):  # 遍歷群組內的行
                for y in range(group_cols):  # 遍歷群組內的列
                    block.append((i + x) * ris_cols + (j + y))  # 計算 index
            group_mapping.append(block)
    print(f"group_mapping: {group_mapping}")

    # 儲存至 CSV
    folder_name = folder_name.removeprefix('./csv/')
    csv_path = f'./generate_RIS/block_wise_phase_grouping/{scenario}/{folder_name}_{count_tmp}'
    os.makedirs(csv_path, exist_ok=True)

    base_filename = f'block_wise_phase_grouping_{args.bits}-bits_{K}UE_{group_rows}x{group_cols}-blocks_{count_tmp}'
    part_index = 0

    def current_file():
        return os.path.join(csv_path, f"{base_filename}_part{part_index}.csv")

    MAX_FILE_SIZE_MB = 40 # 每個檔案的最大大小 (MB)
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

    column_n = [f'Element{i+1}_Phase' for i in range(num_elements)] + \
                [f'UE{i+1}_SINR' for i in range(K)] + \
                [f'UE{i+1}_Datarate' for i in range(K)] + \
                ['Average_SINR', 'Average_Datarate', 'Sum_SINR', 'Sum_Datarate']

    total_combinations = num_phases ** num_blocks
    print(f"總組合數量 (Block-wise Phase Grouping): {total_combinations}")

    Z_blockcomb = torch.ones((1, num_elements), device=args.device)  # Blocking matrix
    print(f'Z_blockcomb: {Z_blockcomb} \nshape: {Z_blockcomb.shape}\n')

    file_path = current_file()
    row_count = 0

    with tqdm(desc=f"Processing num_blocks={num_blocks}", total=total_combinations) as pbar:
        for values in itertools.product(range(num_phases), repeat=num_blocks):
            phase_config = np.zeros(num_elements, dtype=np.float32)
            for group_idx, group in enumerate(group_mapping):
                phase_config[group] = phases_discrete[values[group_idx]]
            print(f'phase_config: {phase_config}, length: {len(phase_config)}')

            theta_radians = torch.tensor(phase_config, dtype=torch.float32).reshape(1, -1).to(device)
            theta_complex = torch.polar(torch.ones_like(theta_radians), theta_radians).to(device)
            Z_theta_blockcomb = Z_blockcomb * theta_complex

            H_blockcomb = channel.get_joint_channel_coefficient(Z_theta_blockcomb)
            W_blockcomb = beamformer.MRT(H_blockcomb)

            # 計算 SINR 和 Datarate
            sinr_linear_blockcomb = channel.SINR(H_blockcomb, W_blockcomb)
            print(f'sinr_linear_blockcomb: {sinr_linear_blockcomb}, shape: {sinr_linear_blockcomb.shape}')
            sinr_db_blockcomb = 10 * torch.log10(sinr_linear_blockcomb).cpu().numpy()
            print(f'sinr_db_blockcomb: {sinr_db_blockcomb}, shape: {sinr_db_blockcomb.shape}')
            datarate_blockcomb = torch.log2(1 + sinr_linear_blockcomb).cpu().numpy()
            print(f'datarate_blockcomb: {datarate_blockcomb}, shape: {datarate_blockcomb.shape}')

            # 記錄數據
            sinr_records = [
                *phase_config,  # 64 個 elements 的 phase 配置
                *sinr_db_blockcomb.flatten(),  # K 個 UE 的 SINR
                *datarate_blockcomb.flatten(),  # K 個 UE 的 Datarate
                sinr_db_blockcomb.mean(),  # 平均 SINR
                datarate_blockcomb.mean(),  # 平均 Datarate
                sinr_db_blockcomb.sum(),   # 總 SINR
                datarate_blockcomb.sum()   # 總 Datarate
            ]

            df_row = pd.DataFrame([sinr_records], columns=column_n)
            write_header = not os.path.exists(file_path)
            df_row.to_csv(file_path, mode='a', header=write_header, index=False)

            row_count += 1
            pbar.update(1)

            # 檢查是否超過檔案大小限制
            if os.path.getsize(file_path) >= MAX_FILE_SIZE_BYTES:
                print(f"[警告] 檔案 {file_path} 超過 {MAX_FILE_SIZE_MB}MB, 開始寫入下一個檔案。")
                part_index += 1
                file_path = current_file()

            # 清除記憶體
            del H_blockcomb, W_blockcomb, theta_radians, theta_complex, Z_theta_blockcomb
            torch.cuda.empty_cache()            # 釋放 GPU 記憶體, 避免 CUDA 記憶體持續累積
            gc.collect()
            print(f'\n')

    print(f'Block-wise Phase Sampling (num_blocks={group_rows}x{group_cols}, results saved to {csv_path}\n')

def known_highSINR_phase_sampling(channel, beamformer, scenario, folder_name, num_elements, K, device, count_tmp, args):
    '''
    基於已知高SINR phase增添擾動擴展探索, 取得鄰近高SINR phase組合, 隨後計算高SINR組合的數量
    '''

    init_display_settings()
    
    print("==================Test 7 (Targeted Perturbation from Known High SINR Phase)===================")    

    # 設定 SINR 閾值
    x = 49  # SINR ≥ x 即為高 SINR
    perturb_samples = 10000000  # 擾動次數
    num_phases = 2 ** args.bits  # 計算總共的 PhaseShift 數量
    phases_discrete = torch.linspace(
        start=0.0,                      # 確保是 float
        end=2 * np.pi * (num_phases - 1) / num_phases,  # 確保不包含 2π
        steps=num_phases,
        dtype=torch.float32             # 設定數據類型
    ).to(args.device)                   # 將數據移至 GPU
    print(f"phases_discrete: {phases_discrete}\n")  # Debug 確認 phase 值

    # 計算完整母體的組合總數
    total_phase_combinations = num_phases ** num_elements  
    print(f"總組合數量 (母體大小): {total_phase_combinations}")

    Z_allcomb = torch.ones((1, num_elements), device=args.device)

    # 計算已知全 0 Phase 的 SINR 和 Datarate
    known_highSINR_phase = np.zeros(num_elements, dtype=np.float32)  # 32 個 element 全為 0
    print(f"計算全 0 Phase 設定的 SINR 和 Datarate...")

    theta_radians = torch.tensor(known_highSINR_phase, dtype=torch.float32).reshape(1, -1).to(args.device)
    theta_complex = torch.polar(torch.ones_like(theta_radians), theta_radians).to(args.device)
    # print(f'theta_radians: {theta_radians}')
    # print(f'theta_complex: {theta_complex}')

    Z_theta = Z_allcomb * theta_complex
    H = channel.get_joint_channel_coefficient(Z_theta)
    W = beamformer.MRT(H)
    sinr_linear = channel.SINR(H, W)
    sinr_db = 10 * torch.log10(sinr_linear).cpu().numpy()
    datarate = torch.log2(1 + sinr_linear).cpu().numpy()

    print(f"全 0 Phase 設定的平均 SINR: {sinr_db.mean():.2f} dB")
    print(f"全 0 Phase 設定的平均 Datarate: {datarate.mean():.2f}")

    # 擾動已知高 SINR Phase 設定
    print(f"開始對已知高 SINR Phase 進行擾動 ({perturb_samples} 次)...")
    perturb_records = []

    with tqdm(total=perturb_samples, desc="Processing Perturbation Samples") as pbar:
        for _ in range(perturb_samples):
            perturbed_config = known_highSINR_phase.copy()

            # 擾動 5~15 個 RIS elements
            num_perturb = np.random.randint(5, 15)
            perturb_indices = np.random.choice(num_elements, num_perturb, replace=False)
            perturbed_config[perturb_indices] = np.random.choice(phases_discrete, num_perturb)

            # 計算擾動後的 SINR
            theta_radians = torch.tensor(perturbed_config, dtype=torch.float32).reshape(1, -1).to(args.device)
            theta_complex = torch.polar(torch.ones_like(theta_radians), theta_radians).to(args.device)

            Z_theta = Z_allcomb * theta_complex
            H = channel.get_joint_channel_coefficient(Z_theta)
            W = beamformer.MRT(H)
            sinr_linear = channel.SINR(H, W)
            sinr_db = 10 * torch.log10(sinr_linear).cpu().numpy()
            datarate = torch.log2(1 + sinr_linear).cpu().numpy()

            # 儲存擾動結果
            perturb_records.append((
                *perturbed_config, 
                *sinr_db.flatten(),
                *datarate.flatten(), 
                sinr_db.mean(),  # 計算所有 UE 的平均 SINR
                datarate.mean()  # 計算所有 UE 的平均 Datarate
            ))

            pbar.update(1)

    column_n = [f'Element{i+1}_Phase' for i in range(num_elements)] + \
                [f'UE{i+1}_SINR' for i in range(K)] + \
                [f'UE{i+1}_Datarate' for i in range(K)] + \
                ['Average_SINR', 'Average_Datarate']

    df_perturb = pd.DataFrame(perturb_records, columns=column_n)

    # 計算 SINR ≥ 50 的組合比例
    sinr_count_sample = (df_perturb['Average_SINR'] >= x).sum()
    sinr_ratio_sample = sinr_count_sample / perturb_samples  # 取樣內的比例

    # 推估完整母體中的 SINR ≥ 50 佔比
    estimated_high_sinr_count = total_phase_combinations * sinr_ratio_sample
    high_sinr_ratio_estimated = estimated_high_sinr_count / total_phase_combinations

    print(f"在擾動取樣內, SINR ≥ {x} 的組合數: {sinr_count_sample}")
    print(f"在擾動取樣內, SINR ≥ {x} 佔比: {sinr_ratio_sample * 100:.6f}%")
    # print(f"該區域在完整母體內的估計 SINR ≥ {x} 佔比: {high_sinr_ratio_estimated * 100:.10f}%")
    print(f"推估的高 SINR Phase 配置總數: {estimated_high_sinr_count:.0f} 組")

    # 儲存到 CSV
    folder_name = folder_name.removeprefix('./csv/')
    csv_path_allcomb = f'./generate_RIS/known_highSINR_phase_sampling/{scenario}/{folder_name}_{count_tmp}'
    os.makedirs(csv_path_allcomb, exist_ok=True)
    filename_allcomb = f'known_highSINR_phase_sampling_{args.bits}-bits_{count_tmp}.csv'
    csv_path_allcomb_addfile = os.path.join(csv_path_allcomb, filename_allcomb)

    df_perturb.to_csv(csv_path_allcomb_addfile, mode='w', header=True, index=False)

    print(f'Targeted Phase Sampling saved to {csv_path_allcomb_addfile}.')

def save_csv_in_parts(df, base_filename, output_folder, column_names, max_file_size=50):
    """
    拆分 DataFrame 並儲存 CSV, 每個檔案大小不超過 max_file_size MB。
    
    參數：
    - df: 需要儲存的 DataFrame
    - base_filename: 輸出的基礎檔名（不含副檔名）
    - output_folder: 儲存的資料夾路徑
    - column_names: 欄位名稱（確保一致）
    - max_file_size: 單個 CSV 最大大小 (單位 MB, 預設 50MB)
    """
    os.makedirs(output_folder, exist_ok=True)
    file_index = 1
    rows_per_file = len(df)  # 預設存整個檔案
    estimated_size = (df.memory_usage(index=True, deep=True).sum() / (1024 * 1024))  # 估計 DataFrame 記憶體大小（MB）

    # 如果 DataFrame 預估大小超過 max_file_size, 則計算適當的拆分行數
    if estimated_size > max_file_size:
        rows_per_file = int(len(df) * (max_file_size / estimated_size))  # 計算適合的分割大小

    print(f"預估總大小: {estimated_size:.2f} MB, 將拆分為多個檔案 (每個最大 {max_file_size} MB)")

    for i in range(0, len(df), rows_per_file):
        part_filename = f"{base_filename}_part{file_index}.csv"
        output_path = os.path.join(output_folder, part_filename)

        # 儲存 CSV, 確保每個檔案都帶有正確的 column_names
        df.iloc[i:i+rows_per_file].to_csv(output_path, mode='w', header=True, index=False, columns=column_names)

        print(f"已儲存: {output_path} (大小約 {os.path.getsize(output_path) / (1024 * 1024):.2f} MB)")
        file_index += 1

def targeted_phase_sampling(channel, beamformer, scenario, folder_name, num_elements, K, device, count_tmp, args):
    '''
    隨機sample一些phase組合, 並且加入擾動擴展探索, 隨後計算高SINR組合的數量
    '''

    init_display_settings()

    count_tmp = count_tmp
    
    print("==================Test 7 (All phase combinations by bits)===================")    
    # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
    # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

    # 設定取樣數量
    x = 45              # SINR>=x 即為高SINR
    N_samples = 500000   # 隨機選擇 50,000 個 RIS phase 組合
    perturb_samples = 100000  # 針對高 SINR 區域, 額外進行 10,000 次 Perturbation
    top_percentage = 0.1  # 取前 10% 高 SINR 組合 

    Z_allcomb = torch.ones((1, num_elements), device=args.device)        # blocking condition, a binary matrix
    print(f'Z_allcomb: {Z_allcomb} \nshape: {Z_allcomb.shape}\n')

    # Define the phase selection list, generating 2^n phase values uniformly distributed within the [0, 2pi] interval
    # 自動根據 bits 產生 Phase Shift 的弧度表
    num_phases = 2 ** args.bits  # 計算總共的 PhaseShift 數量
    phases_discrete = torch.linspace(
        start=0.0,                      # 確保是 float
        end=2 * np.pi * (num_phases - 1) / num_phases,  # 確保不包含 2π
        steps=num_phases,
        dtype=torch.float32             # 設定數據類型
    ).to(args.device)                   # 將數據移至 GPU
    # phase_lookup = np.linspace(0, 2*pi, 2**bits, endpoint=False).tolist()
    print(f'main.py/all_phase_combinations || phases_discrete: {phases_discrete}, len: {len(phases_discrete)}\n')
    # Retrieve and calculate the actual number of RIS elements in the current scenario

    # 計算完整母體的組合總數
    total_phase_combinations = num_phases ** num_elements  
    print(f"總組合數量 (母體大小): {total_phase_combinations}")

    # 先隨機取 N_samples 個組合
    phase_samples = np.random.choice(phases_discrete, size=(N_samples, num_elements))
    # print(f'{phase_samples}, shape: {phase_samples.shape}\n')

    # 記錄 SINR 資料
    sinr_records = []

    # Start the progress bar, with total as total_combinations, reflecting the overall progress of sample combinations
    # Progress bar format: Percentage Progress || current processed combinations/total combinations [elapsed time (min:sec) <estimated time to complete (hr:min:sec), combinations processed per second]
    with tqdm(desc="Processing Phase Combinations", total=N_samples) as pbar:
        
        # Initialize a flag to track if it's the first write (only write the header on the first write)
        first_write = True

        # while True:
        for i in range(N_samples):

            theta_radians = torch.tensor(phase_samples[i], dtype=torch.float32).reshape(1, -1).to(args.device)
            print(f'theta_radians: {theta_radians}, shape: {theta_radians.shape}')
            # Convert radians to complex numbers
            theta_complex = torch.polar(torch.ones_like(theta_radians), theta_radians).to(args.device)
            # print(f'theta_complex: {theta_complex}, shape: {theta_complex.shape}\n')

            Z_theta_allcomb = Z_allcomb * theta_complex
            # print(f'Z_theta_allcomb: {Z_theta_allcomb}')

            H_allcomb = channel.get_joint_channel_coefficient(Z_theta_allcomb)
            W_allcomb = beamformer.MRT(H_allcomb)
            
            # 計算SINR和Datarate
            sinr_linear_allcomb = channel.SINR(H_allcomb, W_allcomb)
            sinr_db_allcomb = 10*torch.log10(sinr_linear_allcomb).cpu().numpy()  
            # print(f"SINR(linear): {sinr_linear_allcomb}, shape: {sinr_linear_allcomb.shape} \n")
            print(f"SINR(dB): {sinr_db_allcomb}, shape: {sinr_db_allcomb.shape}, type: {type(sinr_db_allcomb)} \n")

            # 計算 Shannon 容量公式的 Datarate
            datarate_allcomb = torch.log2(1 + sinr_linear_allcomb).cpu().numpy()

            # 記錄數據
            sinr_records.append((
                *phase_samples[i],  # phase 配置, num_elements 個值
                *sinr_db_allcomb.flatten(),  # 每個 UE 的 SINR, K 個值
                *datarate_allcomb.flatten(),  # 每個 UE 的 Datarate, K 個值
                sinr_db_allcomb.mean(),  # 平均 SINR
                datarate_allcomb.mean()  # 平均 Datarate
            ))

            # Update the progress bar
            pbar.update(1)

    # 轉 DataFrame 並排序
    column_n = [f'Element{i+1}_Phase' for i in range(num_elements)] + \
            [f'UE{i+1}_SINR' for i in range(K)] + \
            [f'UE{i+1}_Datarate' for i in range(K)] + \
            ['Average_SINR', 'Average_Datarate']

    # 轉為 DataFrame
    # phase_columns = [f'Element{i+1}_Phase' for i in range(num_elements)]
    df_sinr = pd.DataFrame(sinr_records, columns=column_n)
    # df_sinr = df_sinr.sort_values(by=[f'UE{i+1}_SINR' for i in range(K)], ascending=False)

    # 2. 計算 SINR ≥ x 的組合佔比
    sinr_count_sample = (df_sinr['Average_SINR'] >= x).sum()
    sinr_ratio_sample = sinr_count_sample / N_samples  # 取樣內的比例

    # 推估母體內的高 SINR 配置數量
    estimated_high_sinr_count = total_phase_combinations * sinr_ratio_sample
    high_sinr_ratio_estimated = estimated_high_sinr_count / total_phase_combinations

    print(f"在取樣內, SINR ≥ {x} 的組合數: {sinr_count_sample}")
    print(f"在取樣內. SINR ≥ {x} 佔比: {sinr_ratio_sample * 100:.6f}%")
    print(f"在完整母體內的估計 SINR ≥ {x} 佔比: {high_sinr_ratio_estimated * 100:.10f}%")
    print(f"推估的高 SINR Phase 配置總數: {estimated_high_sinr_count:.0f} 組")

    # 3. 擾動高 SINR 配置
    top_sinr_samples = df_sinr.nlargest(int(N_samples * top_percentage), 'Average_SINR')

    # # 取出前 10% 高 SINR Phase 設定
    # top_sinr_samples = df_sinr.iloc[:int(N_samples * top_percentage)].copy()
    # print(f"高 SINR 區域: 取出前 {top_percentage * 100:.1f}% 的組合, 共 {len(top_sinr_samples)} 個")

    # 針對高 SINR 組合做 Perturbation
    print(f"開始對高 SINR 區域進行 Perturbation ({perturb_samples} 次)...")
    perturb_records = []
    with tqdm(total=perturb_samples, desc="Processing Perturbation Samples") as pbar:
        for _ in range(perturb_samples):
            base_config = top_sinr_samples.sample(1).iloc[0][:num_elements].values  # 隨機選取一個高 SINR Phase 設定
            perturbed_config = base_config.copy()

            # Perturb 1~5 個 RIS elements
            num_perturb = np.random.randint(5, 15)
            perturb_indices = np.random.choice(num_elements, num_perturb, replace=False)
            perturbed_config[perturb_indices] = np.random.choice(phases_discrete, num_perturb)

            # 計算 Perturbation 後的 SINR
            theta_radians = torch.tensor(perturbed_config, dtype=torch.float32).reshape(1, -1).to(args.device)
            theta_complex = torch.polar(torch.ones_like(theta_radians), theta_radians).to(args.device)

            Z_theta = Z_allcomb * theta_complex
            H = channel.get_joint_channel_coefficient(Z_theta)
            W = beamformer.MRT(H)
            sinr_linear = channel.SINR(H, W)
            sinr_db = 10 * torch.log10(sinr_linear).cpu().numpy()
            datarate = torch.log2(1 + sinr_linear).cpu().numpy()

            # 儲存 Perturbation 結果
            perturb_records.append((
                *perturbed_config, 
                *sinr_db.flatten(), 
                *datarate.flatten(), 
                sinr_db.mean(),  # 這裡計算所有 UE 的平均 SINR
                datarate.mean()  # 這裡計算所有 UE 的平均 Datarate
            ))

            pbar.update(1)

    df_perturb = pd.DataFrame(perturb_records, columns=column_n)

    sinr_count_sample = (df_perturb['Average_SINR'] >= x).sum()
    sinr_ratio_sample = sinr_count_sample / perturb_samples  

    estimated_high_sinr_count = total_phase_combinations * sinr_ratio_sample
    high_sinr_ratio_estimated = estimated_high_sinr_count / total_phase_combinations

    print(f"在擾動取樣內, SINR ≥ {x} 的組合數: {sinr_count_sample}")
    print(f"在擾動取樣內, SINR ≥ {x} 佔比: {sinr_ratio_sample * 100:.6f}%")
    print(f"推估的高 SINR Phase 配置總數: {estimated_high_sinr_count:.0f} 組")

    folder_name = folder_name.removeprefix('./csv/')
    csv_path_allcomb = f'./generate_RIS/targeted_phase_sampling/{scenario}/{folder_name}_{count_tmp}'
    os.makedirs(csv_path_allcomb, exist_ok=True)

    save_csv_in_parts(df_perturb, f'targeted_phase_sampling_{args.bits}-bits_{count_tmp}', csv_path_allcomb, column_n)

    print(f'Targeted Phase Sampling saved to {csv_path_allcomb}')

def generate_all_combinations(a_dim, num_elements, total_combinations):
    """
    使用進制轉換生成所有組合。
    
    Args:
        a_dim: element數量, 會作為組合數量的基數, 如 4^4 次方
        num_elements: RIS element 數量
        total_combinations: 要生成的組合總數
    
    Returns:
        一個包含所有組合的生成器, 每個組合都是長度為 num_elements 的 list
    """
    for index in range(total_combinations):
        combination = []
        temp_index = index
        for _ in range(num_elements):
            combination.append(temp_index % a_dim)  # 計算當前位的值
            temp_index //= a_dim                    # 更新索引
        yield list(reversed(combination))          # 反轉組合順序以符合進制轉換法

def phase_combinations(channel, beamformer, scenario, folder_name, num_elements, K, device, count_tmp, args):
    
    init_display_settings()

    count_tmp = count_tmp
    
    print("==================Test 7 (All phase combinations by bits)===================")    
    # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
    # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

    # Simulate {llcomb_n_timeslot} timeslots, e.g., 1000 timeslots, with multiple parallel environments
    allcomb_n_timeslot = 1                   

    Z_allcomb = torch.ones((allcomb_n_timeslot, num_elements), device=args.device)        # blocking condition, a binary matrix
    print(f'Z_allcomb: {Z_allcomb} \nshape: {Z_allcomb.shape}\n')

    # Define the phase selection list, generating 2^n phase values uniformly distributed within the [0, 2pi] interval
    # 自動根據 bits 產生 Phase Shift 的弧度表
    num_phases = 2 ** args.bits  # 計算總共的 PhaseShift 數量
    phases_discrete = torch.linspace(
        start=0.0,                      # 確保是 float
        end=2 * np.pi * (num_phases - 1) / num_phases,  # 確保不包含 2π
        steps=num_phases,
        dtype=torch.float32             # 設定數據類型
    ).to(args.device)                   # 將數據移至 GPU
    # phase_lookup = np.linspace(0, 2*pi, 2**bits, endpoint=False).tolist()
    print(f'main.py/all_phase_combinations || phases_discrete: {phases_discrete}, len: {len(phases_discrete)}\n')
    # Retrieve and calculate the actual number of RIS elements in the current scenario

    # Sample the number of outputs
    sample = len(phases_discrete) ** num_elements       # 50000

    # Set the total number of combinations to process, limited to the first sample combinations
    total_combinations = min(sample, len(phases_discrete)**num_elements)
    print(f'Total combinations (limited to {sample}): {total_combinations}')
    # Generate the first sample combinations of RIS element phase shifts
    all_combinations = generate_all_combinations(
        a_dim = len(phases_discrete),                 # phase_lookup 的長度作為基數
        num_elements = num_elements,                # RIS 的元素數量
        total_combinations = total_combinations     # 總共要生成的組合數
    )
    for i, comb in enumerate(all_combinations):
        if i >= 100:
            break
        print(comb)

    # Define the number of combinations processed in each batch (e.g., batch_size) and calculate the total number of batches
    batch_size = 1000
    # Set a maximum file size limit of 100MB for each CSV file
    max_file_size = 50 * 1024 * 1024  # 100 >> 50MB, 50 >> 25MB
    file_count = 1          # Document counter, used for file naming

    # Save to CSV
    folder_name = folder_name.removeprefix('./csv/')  # 去掉前綴 './csv/'
    csv_path_allcomb = f'./generate_RIS/all_phase_combinations/{scenario}/{folder_name}_{count_tmp}'
    os.makedirs(csv_path_allcomb, exist_ok=True)
    filename_allcomb = f'all_phase_combinations_{args.bits}-bits_{count_tmp}_part{file_count}.csv'
    csv_path_allcomb_addfile = os.path.join(csv_path_allcomb, filename_allcomb)
    # Clear the old CSV file by opening it in 'w' mode, ensuring previous content is not appended during each execution
    with open(csv_path_allcomb_addfile, 'w') as f:
        pass
    current_file_size = 0
    # Set column name
    column_n = [f'Element{i+1}\'s_Action' for i in range(num_elements)] + \
                [f'Element{i+1}\'s_Phase' for i in range(num_elements)] + \
                [f'UE{i+1}\'s_SINR' for i in range(K)] + \
                [f'UE{i+1}\'s_Datarate' for i in range(K)] + \
                ['Average_SINR', 'Average_Datarate', 'Total_Datarate']
    
    # For Plot
    sinr_values = []
    datarate_values = []

    # Start the progress bar, with total as total_combinations, reflecting the overall progress of sample combinations
    # Progress bar format: Percentage Progress || current processed combinations/total combinations [elapsed time (min:sec) <estimated time to complete (hr:min:sec), combinations processed per second]
    with tqdm(desc="Processing Phase Combinations", total=total_combinations, unit="combination") as pbar:
        
        # Initialize a flag to track if it's the first write (only write the header on the first write)
        first_write = True

        while True:

            # Extract the next batch of combinations
            batch_combinations = [next(all_combinations, None) for _ in range(batch_size)]
            # 移除 None（在生成器結束後會返回 None）
            batch_combinations = [combo for combo in batch_combinations if combo is not None]
            
            if not batch_combinations:          # Exit the loop if no combinations are left
                break
            
            # Start calculating the SINR and data rate for the current batch
            for i, combo in enumerate(batch_combinations):
                # print(f'combo: {combo}\n')

                # Convert the current combination to a tensor and transfer it to CUDA
                theta_allcomb_radians = torch.tensor([phases_discrete[phase] for phase in combo], dtype=torch.float32).reshape(1, -1).to(args.device)                
                # theta_allcomb_radians = torch.tensor(combo, dtype=torch.float32).reshape(1, -1).to(args.device)
                # print(f'theta_allcomb_radians: {theta_allcomb_radians}')

                # Convert radians to complex numbers
                theta_allcomb_complex = torch.polar(torch.ones_like(theta_allcomb_radians, dtype=torch.float32), theta_allcomb_radians).to(args.device)
                # print(f'theta_allcomb_complex: {theta_allcomb_complex}')

                Z_theta_allcomb = Z_allcomb * theta_allcomb_complex
                # print(f'Z_theta_allcomb: {Z_theta_allcomb}')

                H_allcomb = channel.get_joint_channel_coefficient(Z_theta_allcomb)
                W_allcomb = beamformer.MRT(H_allcomb)
                
                # 計算SINR和Datarate
                sinr_linear_allcomb = channel.SINR(H_allcomb, W_allcomb)
                sinr_db_allcomb = 10*torch.log10(sinr_linear_allcomb).cpu().numpy()  
                # print(f"SINR(linear): {sinr_linear_allcomb}, shape: {sinr_linear_allcomb.shape} \n")
                # print(f"SINR(dB): {sinr_db_allcomb}, shape: {sinr_db_allcomb.shape}, type: {type(sinr_db_allcomb)} \n")

                if sinr_linear_allcomb is not None:
                    # Calculate channel capacity based on Shannon's theorem
                    # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
                    channel_capacity_allcomb = torch.log2(1 + sinr_linear_allcomb)
                    # Assume each symbol can carry 1 bit
                    # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
                    datarate_allcomb = channel_capacity_allcomb
                else:
                    datarate_allcomb = 0
                # print(f"Datarate: {datarate_allcomb}, shape: {datarate_allcomb.shape}, type: {type(datarate_allcomb)} \n")

                # Calculate the Avg. SINR, Total Datarate, and Avg. Datarate for UE 1~16.
                avg_sinr_allcomb = np.mean(sinr_db_allcomb[:K])
                avg_datarate_allcomb = datarate_allcomb[:K].mean().cpu().item()
                total_datarate_allcomb = datarate_allcomb[:K].sum().cpu().item()

                # Ensure combo is correctly saved and separates SINR and datarate
                row = list(combo) + list([phases_discrete[phase] for phase in combo])+list(sinr_db_allcomb.flatten()) + list(datarate_allcomb.flatten().cpu().numpy()) + [avg_sinr_allcomb, avg_datarate_allcomb, total_datarate_allcomb]
                print(f'INSERT ROW: {row}\n')

                # 這裡做間隔存檔, 每 5 筆存一次
                n = 5
                if i % n == 0:  
                    # Convert row to DataFrame format
                    allcomb_row_df = pd.DataFrame([row], columns=column_n)

                    # Check file size and switch files if necessary
                    current_file_size += allcomb_row_df.memory_usage(deep=True).sum()
                    if current_file_size > max_file_size:
                        file_count += 1
                        filename_allcomb = f'all_phase_combinations_{args.bits}-bits_{count_tmp}_part{file_count}.csv'
                        csv_path_allcomb_addfile = os.path.join(csv_path_allcomb, filename_allcomb)
                        with open(csv_path_allcomb_addfile, 'w') as f:
                            pass
                        current_file_size = 0
                        first_write = True  # A new file needs to write headers

                    # Save data to the current file (append mode)
                    allcomb_row_df.to_csv(csv_path_allcomb_addfile, mode='a', header=first_write, index=False)

                    # For Plot
                    sinr_values.append(sinr_db_allcomb.flatten())
                    datarate_values.append(datarate_allcomb.flatten())
                    
                    # Keep the header only during the first write, and skip writing the header in subsequent writes
                    first_write = False

            # Update the progress bar
            pbar.update(len(batch_combinations))

    print(f'Test 7 allcomb: all_phase_combinations saved to {csv_path_allcomb_addfile}.')

    # # ========================Plot: Plot Configuration and Data Sampling========================
    # # Calculate number of data points; auto-adjust figure width: min 20, max 150 based on data points
    # num_points = len(sinr_values)
    # fig_width = max(20, min(150, num_points / 100))

    # # Dynamically adjust figure height based on SINR range, constrained between 5 and 12
    # sinr_range = max([x.mean() for x in sinr_values]) - min([x.mean() for x in sinr_values])
    # fig_height = max(5, min(12, sinr_range * 5))

    # # Sample data for display if too large, with a maximum of 5000 points shown
    # max_points = 5000
    # if num_points > max_points:
    #     indices = np.linspace(0, num_points - 1, max_points).astype(int)
    #     sinr_values_sampled = [sinr_values[i] for i in indices]
    #     datarate_values_sampled = [datarate_values[i] for i in indices]
    # else:
    #     sinr_values_sampled = sinr_values
    #     datarate_values_sampled = datarate_values

    # # Auto-adjust X-axis ticks: show a tick every 10% of the data, at least one tick
    # xticks_interval = max(1, len(sinr_values_sampled) // 10)

    # # ============Plot: SINR============
    # sinr_values_sampled = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in sinr_values_sampled]
    # plt.figure(figsize=(fig_width, fig_height), dpi=300)
    # plt.plot(
    #     range(1, len(sinr_values_sampled) + 1), 
    #     [x.mean() for x in sinr_values_sampled],                # Conversion is no longer needed here as it's already NumPy
    #     label='SINR (dB)', 
    #     color='brown'
    # )

    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # plt.xlabel('Phase Combination', fontsize=12)
    # plt.ylabel('SINR (dB)', fontsize=12)
    # plt.title('Each phase combination\'s SINR', fontsize=14)
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(range(1, len(sinr_values_sampled) + 1, xticks_interval), fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.tight_layout()

    # sinr_fig_name = f'SINR_{args.bits}-bits_{count_tmp}.png'
    # allcomb_sinr_plot_path = os.path.join(csv_path_allcomb, sinr_fig_name)
    # plt.savefig(allcomb_sinr_plot_path)

    # print(f'Test 7 allcomb: SINR plot saved to {allcomb_sinr_plot_path}.')

    # # ============Plot: Datarate============
    # datarate_values_sampled = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in datarate_values_sampled]
    # plt.figure(figsize=(fig_width, fig_height), dpi=300)
    # plt.plot(
    #     range(1, len(datarate_values_sampled) + 1), 
    #     [x.mean() for x in datarate_values_sampled],                # Conversion is no longer needed here as it's already NumPy
    #     label='Datarate', 
    #     color='blue'
    # )

    # plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    # plt.xlabel('Phase Combination', fontsize=12)
    # plt.ylabel('Datarate', fontsize=12)
    # plt.title('Each phase combination\'s Datarate', fontsize=14)
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(range(1, len(datarate_values_sampled) + 1, xticks_interval), fontsize=10)
    # plt.yticks(fontsize=10)
    # plt.tight_layout()

    # datarate_fig_name = f'Datarate_{args.bits}-bits_{count_tmp}.png'
    # allcomb_datarate_plot_path = os.path.join(csv_path_allcomb, datarate_fig_name)
    # plt.savefig(allcomb_datarate_plot_path)

    # print(f'Test 7 allcomb: Datarate plot saved to {allcomb_datarate_plot_path}.')

def channel_beam(channel, beamformer, scenario, folder_name, num_elements, h_ris, w_ris, args, d_ris_elem):
    
    init_display_settings()
    plt.close('all')  # 關閉所有 Matplotlib Figure, 清理內存

    print("==================Test 7 (All phase combinations by bits)===================")    
    # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
    # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

    # Simulate {llcomb_n_timeslot} timeslots, e.g., 1000 timeslots, with multiple parallel environments
    n_env = 2 ** (args.bits * 2)                   
    print(f'n_env: {n_env}')
    n_iter = 1

    RIS_beambook = []
    for theta, phi in product(np.linspace(0, 2 * np.pi, 2**args.bits), np.linspace(0, 2 * np.pi, 2**args.bits)):
        RIS_beambook.append((theta, phi))
    print(f'RIS_beambook: {RIS_beambook}')

    # Z_allcomb = torch.ones((len(RIS_beambook), num_elements), device=args.device)
    Z_allcomb = torch.ones((n_env, num_elements), device=args.device)        # blocking condition, a binary matrix
    print(f'Z_allcomb: {Z_allcomb} \nshape: {Z_allcomb.shape}\n')


    Theta_allcomb = torch.polar(torch.ones_like(Z_allcomb), torch.zeros_like(Z_allcomb))  # phase shift
    print(f'Theta_allcomb: {Theta_allcomb} \nshape: {Theta_allcomb.shape}\n')

    # for裡在做朝不同的方向亂反射, 只是要全部跑過一輪, 像是逐個bit調phase的概念 (地毯式跑過所有phase)
    for idx_beam, (theta, h) in enumerate(RIS_beambook):
        print(f'idx_beam: {idx_beam}, theta: {theta}, h: {h}')
        x_steer_vec = torch.polar(torch.ones(h_ris), torch.Tensor(2 * np.pi * d_ris_elem / channel.wavelength * np.sin(theta) * np.sin(phi) * np.arange(h_ris)))
        print(f'x_steer_vec: {x_steer_vec} \nshape: {x_steer_vec.shape}')
        y_steer_vec = torch.polar(torch.ones(w_ris), torch.Tensor(2 * np.pi * d_ris_elem / channel.wavelength * np.sin(theta) * np.cos(phi) * np.arange(w_ris)))
        print(f'y_steer_vec: {y_steer_vec} \nshape: {y_steer_vec.shape}')
        Theta_allcomb[idx_beam] = torch.kron(y_steer_vec, x_steer_vec).flatten()
        print(f'Theta_allcomb: {Theta_allcomb} \nshape: {Theta_allcomb.shape}\n')
    Theta_allcomb.to(device=args.device)
    print(f'Theta_allcomb: {Theta_allcomb}')

    B_BS, B_Sk, B_Bk = channel.getBlockMats()  # Blocking condition
    D_BS, D_Sk, D_Bk = channel.getDistMats()  # Distances (m)

    # channel.update_environment()
    Pr_BS_RIS_UE_avg = 0
    Pr_BS_UE_avg = 0
    Pr_UE_avg = 0
    SINR_avg = 0
    for i in tqdm(range(n_iter)):
        # channel.update_channel(alpha_los=2, alpha_nlos=4)

        h_Bk, h_BS, h_Sk, AWGN_norm = channel.get_channel_coefficient()
        H = channel.get_joint_channel_coefficient(Z_allcomb * Theta_allcomb)
        W = beamformer.MRT(H)
        Pr_BS_RIS_UE = torch.abs(torch.einsum("kn,zn,nm,zkm->zk", h_Sk, Z_allcomb * Theta_allcomb, h_BS, W)) ** 2
        Pr_BS_UE = torch.abs(torch.einsum("km,zkm->zk", h_Bk, W)) ** 2
        Pr_UE = (20e6 * channel.Pt_density_BS) * torch.abs(torch.einsum("kn,zn,nm,zkm->zk", h_Sk, Z_allcomb * Theta_allcomb, h_BS, W) + torch.einsum("km,zkm->zk", h_Bk, W)) ** 2
        SINR = channel.SINR(H, W)

        Pr_BS_RIS_UE_avg += Pr_BS_RIS_UE
        Pr_BS_UE_avg += Pr_BS_UE
        Pr_UE_avg += Pr_UE
        SINR_avg += SINR
        # print(f"BS-UE: {10 * torch.log10(torch.abs(torch.sum(h_Bk))).item()} dB")     # 這行不是宣逸寫的, 不確定可以這樣算, 要問宣逸
        print(f"BS-RIS: {10 * torch.log10(torch.abs(torch.sum(h_BS))).item()} dB")
        print(f"RIS-UE: {10 * torch.log10(torch.abs(torch.sum(h_Sk))).item()} dB")

    Pr_BS_RIS_UE_avg = 10 * torch.log10(Pr_BS_RIS_UE_avg / n_iter)
    Pr_BS_UE_avg = 10 * torch.log10(Pr_BS_UE_avg / n_iter)
    Pr_UE_avg = 10 * torch.log10(Pr_UE_avg / n_iter)
    SINR_avg = 10 * torch.log10(SINR_avg / n_iter)

    folder_name = folder_name.removeprefix('./csv/')  # 去掉前綴 './csv/'
    print(f'folder_name: {folder_name}')
    dir = f"./generate_RIS/channel_beam/{scenario}/{folder_name}"
    if not os.path.isdir(dir):
        os.makedirs(dir, exist_ok=True)

    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].plot(range(n_env), Pr_BS_RIS_UE_avg.cpu().numpy(), color="b", label="BS-RIS-UE")
    ax[0, 0].plot(range(n_env), Pr_BS_UE_avg.cpu().numpy(), color="r", label="BS-UE")
    # ax.hlines(Pr_BS_UE.cpu().numpy(), 0, n_env - 1, colors="r", label="BS-UE")
    ax[0, 0].legend(loc="best")
    ax[0, 0].set_title("Channel Gain")
    ax[0, 0].set_xlabel("beam number")
    ax[0, 0].set_ylabel("Channel gain (dB)")
    ax[0, 0].grid()

    ax[0, 1].plot(range(n_env), Pr_BS_RIS_UE_avg.cpu().numpy() - Pr_BS_UE_avg.cpu().numpy(), label="BS-RIS-UE/BS-UE")
    ax[0, 1].set_title("BS-RIS-UE/BS-UE")
    ax[0, 1].set_xlabel("beam number")
    ax[0, 1].set_ylabel("Ratio (dB)")
    ax[0, 1].grid()

    ax[1, 0].plot(range(n_env), Pr_UE_avg.cpu().numpy() + 30, label="Pr")
    ax[1, 0].set_title("Received power")
    ax[1, 0].set_xlabel("beam number")
    ax[1, 0].set_ylabel("Pr (dBm)")
    ax[1, 0].grid()

    ax[1, 1].plot(range(n_env), SINR_avg.cpu().numpy(), label="SINR")
    ax[1, 1].set_title("SINR")
    ax[1, 1].set_xlabel("beam number")
    ax[1, 1].set_ylabel("SINR (dB)")
    ax[1, 1].grid()

    fig.suptitle(f"RIS size: {h_ris * channel.wavelength / 2 * 100:.2f} cm x{w_ris * channel.wavelength / 2 * 100:.2f} cm")
    fig.tight_layout()

    fig.savefig(os.path.join(dir, "channel_gain.png"))

    channel.plot_block_cond(dir)
    channel.show(dir)
    channel.plot_SINR(dir, SINR[0])  # rarely used

def main(args):

    init_display_settings()

    # === 取得 scenario config ==
    wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode, scale, ris_height, obs3 = scenario_configs(scenario)

    num_elements = ris_size[0] * ris_size[1]

    # For parameters JSON 紀錄: 轉換 obstacle 物件為 dict
    def extract_obstacle_info(obs):
        if obs is None:
            return None
        return {
            "pos": list(getattr(obs, "pos", [])),
            "size": list(getattr(obs, "size", [])),
            "height": getattr(obs, "height", None),
            "rotate": getattr(obs, "rotate", None)
        }

    # For parameters JSON 紀錄: 建立 extra_config 字典, 將 utils.py 的變數加入
    extra_config = {
        "wavelength": wavelength,
        "scenario": scenario,
        "scale": scale,
        "area": list(area),
        "BS_pos": list(BS_pos),
        "ris_size": list(ris_size),
        "ris_norm": list(ris_norm),
        "ris_center": list(ris_center),
        "ris_height": ris_height,
        "centers": [list(c) for c in centers],
        "std": std,
        "K": K,
        "MU_mode": MU_mode,
        "obs3": extract_obstacle_info(obs3),
        "obstacles": [extract_obstacle_info(obs) for obs in obstacles], 
    }

    print(f"fc: {3e8 / wavelength * 1e-9:.4f} GHz")
    print(f'Device: {args.device}')
    print(f'Scenario: {scenario}')
    print(f"Seed: {args.seed}")
    # 如果 args.seed 是 None, 則隨機產生一個 seed
    if args.seed is None:
        args.seed = random.randint(0, 2025)  # 產生 0 到 2025 之間的隨機整數

    num_phases, phases_discrete = get_phase_config(args)
    print(f"agent.py/init_agent || num_phases: {num_phases}")
    print(f"agent.py/init_agent || phases_discrete: {phases_discrete}")

    # 產生 RIS 群組
    num_groups, group_mapping = create_group_mapping(ris_size, args.group_size)
    print(f"main.py || num_groups: {num_groups}")
    print(f"main.py || group_mapping: {group_mapping}")

    channel = Channel(
        wavelength = wavelength,
        d_ris_elem = d_ris_elem,
        ris_center = ris_center,
        ris_norm = ris_norm,      # z=0
        ris_size = ris_size,
        area = area,
        BS_pos = BS_pos,
        M = M,
        K = K,
        device = args.device,
        MU_dist = MU_mode,
        rand_seed = args.seed,
        args = args,
    )
    beamformer = Beamformer(device = args.device)

    channel.config_scenario(
        centers_ = centers,
        obstacles = obstacles,
        std_radius = std,
        scenario_name = scenario
    )
    channel.create()

    state_dim = num_elements + K                                            # 32*32個element(1024) + 50個UE的SINR
    action_dim = num_groups * num_phases  # Action 由 (選擇群組, 選擇 Phase) 共同決定

    folder_name = './csv/' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '_seed_' + str(args.seed)
    with open('./csv/train_folder_name.txt', 'w') as f:
        f.write(folder_name)
    if args.save_data:
        save_csv = SaveCSV(folder_name=folder_name)

    print("======================================== Start Training =========================================")

    episode = 1                                                             # 初始化 episode(=timeslot), 尚桓設2

    print(f"=========================== Initial phase state ============================")
    initial_phase_state, initial_sinr_linear, initial_sinr_db, datarate = Phase_state(episode, channel, beamformer, folder_name, num_elements, group_mapping, args)  # 修改 Phase_state 以接受 channel
    # print(f"main.py || initial_phase_state[:] for timeslot {episode}: {initial_phase_state[:]}")
    # print(f"main.py || initial_sinr_linear[:] for timeslot {episode}: {initial_sinr_linear[:]}")
    # print(f"main.py || initial_sinr_db[:] for timeslot {episode}: {initial_sinr_db[:]}")
    # print(f"main.py || initial_datarate[:] for timeslot {episode}: {datarate[:]}\n")

    DDPGAgt = DDPGAgent(K, state_dim, num_groups, num_phases, args.neuron, args)  # 初始化 DDPG

    while episode < args.max_episodes:
    # while episode < 2:        # for DEBUG

        print(f"=================================== Start ==================================")
        env = RISEnv(episode, channel, beamformer, num_elements, num_groups, num_phases, group_mapping, phases_discrete, args)
        # print(f"main.py || env: {env}")

        step = 0                                                            # 初始化步驟計數器 step
        accum_reward = []                                                   # 初始化累積獎勵 accum_reward
        done = []

        for agent in range(1):                                        # agent
            temp_step = 0
            # print(f"step: {step}")
            # print(f"agent: {agent}")
            # print(f"num_elements: {num_elements}")

            state_before, sinr_before_linear, sinr_before_db, datarate_before = env.reset(episode, agent, num_elements, channel, beamformer, group_mapping, folder_name, args)
            # print(f"main.py/Training || state_before: {state_before}, type: {type(state_before)}")    
            # print(f'main.py/Training || sinr_before_linear: {sinr_before_linear}, type: {type(sinr_before_linear)}')
            # print(f'main.py/Training || sinr_before_db: {sinr_before_db}, type: {type(sinr_before_db)}')
            # print(f"main.py/Training || UE Datarate BEFORE adjustment: {datarate_before}, type: {type(datarate_before)}")

            # exit()      # test drawn_scenario

            while temp_step < args.episode_length:

                print(f"==================== RIS Element Phase Adjustment Start ====================")
                print(f'folder_name: {folder_name}')
                print(f"temp_step: {temp_step}")

                action = DDPGAgt.select_action(state_before, num_elements, eval=False)  # DDPG 選擇動作
                # print(f"main.py/Training || choose_action returned: {action}, type: {type(action)}\n")

                next_state, reward_after, done, sinr, sinr_db, curr_state = env.step(action, agent, num_elements)
                # print(f"main.py || done before storing: {done}, type: {type(done)}, shape: {done.shape if isinstance(done, torch.Tensor) else 'N/A'}")
                # print(f"main.py || next_state: {next_state}, type: {type(next_state)}")

                # print(f"main.py || reward_after: {reward_after}, type: {type(reward_after)}")
                # print(f"main.py || done: {done}, type: {type(done)}")
                # print(f"main.py || sinr: {sinr}, type: {type(sinr)}")
                # print(f"main.py || sinr_db: {sinr_db}, type: {type(sinr_db)}")
                # print(f"main.py || curr_state: {curr_state}, type: {type(curr_state)}")

                reward = reward_after

                # 將當前 episode (timexlot) 中 RIS 選擇的動作、reward、state 等資訊存入 memory, 用來累積訓練資料, 等到 batch size 足夠後, PPO 會執行更新
                DDPGAgt.memory.add(state_before, action, reward, next_state, done)

                # # DEBUG: sample ReplayBuffer data
                # if len(DDPGAgt.memory) > 0:  # 確保有數據才能 sample
                #     sampled_state, sampled_action, sampled_reward, sampled_next_state, sampled_done = DDPGAgt.memory.sample(batch_size=1)

                #     print("Sampled state:", sampled_state)
                #     print("Sampled action:", sampled_action)
                #     print("Sampled reward:", sampled_reward)
                #     print("Sampled next state:", sampled_next_state)
                #     print("Sampled done:", sampled_done)
                # else:
                #     print("ReplayBuffer is empty! No data to sample.")

                step += 1
                next_state = next_state.cpu().numpy()                       # 如果 reward 是 torch.Tensor, 轉換為 numpy 陣列
                # print(f"next_state: {next_state}")

                if K == 1:
                    accum_reward = reward.cpu().numpy()                                   # 如果reward只有一個UE, reward=accum_reward
                    accum_datarate_before = datarate_before.cpu().numpy()
                else:
                    if isinstance(reward, torch.Tensor):
                        accum_reward = reward.sum().cpu().numpy()
                    else:
                        accum_reward = np.sum(reward)                    
                    
                    if isinstance(datarate_before, torch.Tensor):
                        accum_datarate_before = datarate_before.sum().cpu().numpy()
                    else:
                        accum_datarate_before = np.sum(datarate_before)
                
                print(f"main.py/Training || Datarate AFTER adjustment(accum_reward): {accum_reward} \n")

                # start training, 如果replay_buffer的長度大於batch_size, 則進行訓練
                # 每當 memory 裡面有至少 batch_size 個就執行訓練
                # 如果 batch_size = 256, 但 buffer 內部存量還沒累積到足夠的有效經驗, 這可能會過早訓練, 導致學習效果不佳
                # 通常做法是等 buffer 內至少有 10% ~ 20% 滿了才開始訓練
                if len(DDPGAgt.memory) >= args.batch_size:
                    DDPGAgt.train(episode, temp_step)

                state_before = next_state

                # 把 accum_reward 轉換, 確保他是數值
                reward_value = float(np.sum(accum_reward))
                # print(f"main.py/Training || Reward(accum_reward): {reward_value}, type: {type(reward_value)}")

                # 取得 Loss 和 Value
                critic_loss, actor_loss = DDPGAgt.getLoss()
                # print(f"critic_loss: {critic_loss}, type: {type(critic_loss)}")
                # 確保 critic_loss 和 actor_loss 是純數值
                if critic_loss is None or actor_loss is None:
                    critic_loss_value = 0.0
                    actor_loss_value = 0.0
                else:
                    critic_loss_value = float(critic_loss.item()) if isinstance(critic_loss, torch.Tensor) else float(critic_loss)
                    actor_loss_value = float(actor_loss.item()) if isinstance(actor_loss, torch.Tensor) else float(actor_loss)

                # critic_loss_value = float(critic_loss) if critic_loss is not None else 0.0
                # actor_loss_value = float(actor_loss) if actor_loss is not None else 0.0

                print("[Timeslot %05d, Agent %05d] Reward %3.12f Critic Loss %3.12f Actor Loss %3.12f\n" % \
                    (episode, agent+1, reward_value, critic_loss_value, actor_loss_value))

                # save data to csv
                if args.save_data:
                    save_csv.insert(episode, agent+1, step, temp_step+1, float(np.sum(accum_reward)), critic_loss_value, actor_loss_value, action)
                    # save_csv.insert(episode, agent+1, step, temp_step+1, np.sum(accum_reward).item(), critic_loss, actor_loss, action)

                temp_step += 1

        episode += 1        # 相當於timeslot+1, 換下一個timeslot

        # ---------------------Update UE position for each timeslot---------------------
        # update_ue_env = channel.update_environment()
        # print(f"Update UE position: {update_ue_env}")

        # Save Model
        if args.save_model:
            print("\n\nModel Save : " + args.log_dir)

            try:
                # 取得完整的 DDPG Checkpoint
                checkpoint = DDPGAgt.getCheckpoint()

                # 儲存模型
                network.save_model(checkpoint, args.log_dir)
                
                # # 從 DDPGAgt 取得 Actor、Critic、Optimizer 以及 Loss
                # ck_actor, ck_critic, ck_actor_target, ck_critic_target, \
                # ck_actor_optimizer, ck_critic_optimizer, ck_loss = DDPGAgt.getCheckpoint()

                # # 儲存完整的 DDPG 模型
                # network.save_model(
                #     ck_actor, ck_critic, ck_actor_target, ck_critic_target,
                #     ck_actor_optimizer, ck_critic_optimizer, ck_loss, args.log_dir
                # )
            except Exception as e:
                print(f"Model saving failed: {e}")

    print("======================================== Start Testing =========================================")

    save_csv_testing = SaveCSV_testing(folder_name=folder_name)

    testing_init_channel = channel.get_channel_coefficient()
    # print(f"main.py/Testing || Get \"None\" Channel Coefficient: {testing_init_channel}")
    # channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)
    # testing_init_update_channel_env = channel.get_channel_coefficient()
    # print(f"main.py/Testing || Update UE position[0][:1]: {testing_init_update_channel_env[0][:1]}...\n")    

    # Z = torch.ones((1, num_elements), device=args.device)                # blocking condition, a binary matrix

    test_ratio = args.max_episodes + math.floor(args.max_episodes / 4)
    count_tmp = 0

    # 如果檔案存在且是第一次寫入, 則先刪除檔案表示重新開始
    # print(f"folder_name: {folder_name}")
    state_csv_file_path_before = os.path.join(folder_name, "RL_testing_before.csv")
    state_csv_file_path_after = os.path.join(folder_name, "RL_testing_after.csv")
    if os.path.exists(state_csv_file_path_before):
        os.remove(state_csv_file_path_before)
    if os.path.exists(state_csv_file_path_after):
        os.remove(state_csv_file_path_after)

    for test_episode in range(args.max_episodes, test_ratio, 1):  # 對於測試數據範圍內的每個數據, 1/4

        accum_reward = []
        
        count_tmp += 1

        env = RISEnv(test_episode, channel, beamformer, num_elements, num_groups, num_phases, group_mapping, phases_discrete, args)

        for agent in range(1):        # agent, 因為我一次是調整整片RIS面板, 在Phase_state中已經會計算四個element phase, 因此這裡不需要再重複計算

            state_before, sinr_before_linear, sinr_before_db, datarate_before = env.reset(test_episode, agent, num_elements, channel, beamformer, group_mapping, folder_name, args)
            # print(f"main.py/Testing || Test_state: {state_before}") 
            # print(f"main.py/Testing || Test_sinr_before_linear: {sinr_before_linear}, shape: {sinr_before_linear.shape}")
            print(f"main.py/Testing || SINR (dB) BEFORE adjustment:: {sinr_before_db}\n")
            print(f"main.py/Testing || Datarate BEFORE adjustment: {datarate_before}\n")
            
            sinr_before_linear_flat = [
                # debug_and_yield(item)        # test: check for DEBUG
                round(item, 8) for x in sinr_before_linear                          # round(item, 8): 每個數值四捨五入到小數點後 8 位
                for item in (x.cpu().numpy().flatten() if isinstance(x, torch.Tensor) else x.flatten())
            ]
            sinr_before_db_flat = [
                # debug_and_yield(item)        # test: check for DEBUG
                round(item, 8) for x in sinr_before_db                              # round(item, 8): 每個數值四捨五入到小數點後 8 位
                for item in (x.flatten() if isinstance(x, np.ndarray) else x)
            ]
            print(f"main.py/Testing || SINR (dB) BEFORE adjustment: {sinr_before_db_flat}")
            # print(f"main.py/Testing || Datarate BEFORE adjustment: {datarate_before}\n")

            # action_test, log_prob_test, value_test = DDPGAgt.select_action(state_before)                # 在每個 episode 中執行動作, 更新狀態和獎勵
            action_test = DDPGAgt.select_action(state_before, num_elements, eval=True)                # 在每個 episode 中執行動作, 更新狀態和獎勵
            # action_test = DDPGAgt.select_action(state_before, args.exploration_noise)                # 在每個 episode 中執行動作, 更新狀態和獎勵
            next_state, reward_after, done, sinr, sinr_db, curr_state = env.step(action_test, agent, num_elements)
            print(f"main.py/Testing || action_test: {action_test}")
            # print(f"main.py/Testing || next_state: {next_state}")
            # print(f"main.py/Testing || curr_state: {curr_state}")
            # print(f"main.py/Testing || SINR (linear) AFTER adjustment: {sinr}")
            print(f"main.py/Testing || SINR (dB) AFTER adjustment: {sinr_db}")
            
            if sinr_db.size == 1:
                sinr_after_db_show = sinr_db.item()
            else:
                sinr_after_db_show = sinr_db.tolist()
            print(f"main.py/Testing || SINR (dB) AFTER adjustment (list): {sinr_after_db_show}")

            # 使用本次datarate與action後的datarate差距作為reward
            # reward = (reward_after - datarate_before)
            # 調整後的datarate作為reward
            reward = reward_after 
            print(f"main.py/Testing || reward: {reward}")

            if K == 1:
                accum_reward = reward.cpu().numpy()                                   # 如果reward只有一個UE, reward=accum_reward
                accum_datarate_before = datarate_before.cpu().numpy()
            else:
                if isinstance(reward, torch.Tensor):
                    reward = reward.squeeze().tolist()
                    accum_reward = sum(reward)
                    # accum_reward = reward.sum().cpu().numpy()
                else:
                    accum_reward = np.sum(reward)                    
                
                if isinstance(datarate_before, torch.Tensor):
                    datarate_before = datarate_before.squeeze().tolist()
                    accum_datarate_before = sum(datarate_before)
                else:
                    accum_datarate_before = np.sum(datarate_before)
            # print(f"main.py/Testing || datarate_before (in testing): {datarate_before} \n")
            # print(f"main.py/Testing || accum_reward (in testing) AFTER adjustment: {accum_reward} \n")

            # accum_reward_show = accum_reward.item()
            print(f"main.py/Testing || Datarate AFTER adjustment: {accum_reward} \n")

            # 將 Tensor 轉換為 list
            # next_state_list = next_state.tolist()
            # print(f"main.py/Testing || next_state_list: {next_state_list}")
            curr_state_list = curr_state.tolist()

            # save testing before results
            save_testing_results(sinr_before_db_flat, accum_datarate_before, datarate_before, state_before[:num_elements], state_csv_file_path_before, extra_config, args)
            # save testing after results
            save_testing_results(sinr_after_db_show, accum_reward, reward, curr_state_list[:num_elements], state_csv_file_path_after, extra_config, args)

            # print results and save to csv
            print("[Timeslot %05d, Agent %05d] Reward %3.12f Action %s \n" % \
                (test_episode, agent+1, accum_reward, action_test))
            save_csv_testing.insert(test_episode, agent+1, accum_reward, action_test)

        # Call channel_beam for each timeslot
        # channel_beam(
        #     channel=channel,
        #     beamformer=beamformer,
        #     scenario=scenario,
        #     folder_name=folder_name,
        #     num_elements=num_elements,
        #     K=K,
        #     device=args.device,
        #     count_tmp=count_tmp
        # )

        # ---------------------Update UE position for each timeslot---------------------
        # update_ue_env = channel.update_environment()
        # print(f"Update UE position: {update_ue_env}")

    # # DEBUG: BEAM
    # channel_beam(
    #     channel=channel,
    #     beamformer=beamformer,
    #     scenario=scenario,
    #     folder_name=folder_name,
    #     num_elements=num_elements,
    #     h_ris=ris_size[0],
    #     w_ris=ris_size[1],
    #     args=args,
    #     d_ris_elem=d_ris_elem
    # )

    # # 將RIS元素分成G個群組, 每組共享相同的phase, 遍歷所有組合, 計算每種組合的SINR和Datarate
    # block_wise_phase_grouping(
    #     channel=channel,
    #     beamformer=beamformer,
    #     scenario=scenario,
    #     folder_name=folder_name,
    #     num_elements=num_elements,
    #     K=K,
    #     device=args.device,
    #     count_tmp=count_tmp,
    #     args=args
    # )

    # # 基於已知高SINR phase增添擾動擴展探索, 取得鄰近高SINR phase組合, 隨後計算高SINR組合的數量
    # known_highSINR_phase_sampling(
    #     channel=channel,
    #     beamformer=beamformer,
    #     scenario=scenario,
    #     folder_name=folder_name,
    #     num_elements=num_elements,
    #     K=K,
    #     device=args.device,
    #     count_tmp=count_tmp,
    #     args=args
    # )

    # # 隨機sample一些phase組合, 並且加入擾動擴展探索, 隨後計算高SINR組合的數量
    # targeted_phase_sampling(
    #     channel=channel,
    #     beamformer=beamformer,
    #     scenario=scenario,
    #     folder_name=folder_name,
    #     num_elements=num_elements,
    #     K=K,
    #     device=args.device,
    #     count_tmp=count_tmp,
    #     args=args
    # )

    # # DEBUG: NO UPDATE
    # phase_combinations(
    #     channel=channel,
    #     beamformer=beamformer,
    #     scenario=scenario,
    #     folder_name=folder_name,
    #     num_elements=num_elements,
    #     K=K,
    #     device=args.device,
    #     count_tmp=count_tmp,
    #     args=args
    # )

    # # No RIS
    # noris(
    #     channel=channel, 
    #     beamformer=beamformer, 
    #     scenario=scenario, 
    #     num_elements=num_elements, 
    #     K=K,
    #     device=args.device
    # )


def save_testing_results(sinr_db, sum_datarate, each_datarate, action_phase, state_csv_file_path, extra_config, args):

    # print(f'datarate: {datarate}')

    init_display_settings()

    data = {}

# --------- Sum Datarate ---------
    if isinstance(sum_datarate, torch.Tensor):
        sum_datarate = sum_datarate.detach().cpu().item()
    elif isinstance(sum_datarate, np.ndarray):
        sum_datarate = float(sum_datarate.flatten()[0])
    elif isinstance(sum_datarate, list):
        sum_datarate = float(np.sum(sum_datarate))
    data["Sum Datarate"] = [round(sum_datarate, 8)]

    # --------- Each UE's Datarate ---------
    if isinstance(each_datarate, torch.Tensor):
        reward_values = each_datarate.detach().cpu().numpy().flatten().tolist()
    elif isinstance(each_datarate, np.ndarray):
        reward_values = each_datarate.flatten().tolist()
    elif isinstance(each_datarate, list):
        reward_values = each_datarate
    else:
        reward_values = [float(each_datarate)]

    for i, val in enumerate(reward_values):
        data[f'UE{i+1}_Datarate'] = [round(float(val), 8)]

    # 加入 Average Datarate
    avg_datarate = float(np.mean(reward_values))
    data["Average Datarate"] = [round(avg_datarate, 8)]

    # --------- Each UE's SINR ---------
    if isinstance(sinr_db, torch.Tensor):
        sinr_values = sinr_db.detach().cpu().numpy().flatten().tolist()
    elif isinstance(sinr_db, np.ndarray):
        sinr_values = sinr_db.flatten().tolist()
    elif isinstance(sinr_db, list):
        sinr_values = sinr_db
    else:
        sinr_values = [float(sinr_db)]

    for i, val in enumerate(sinr_values):
        data[f'UE{i+1}_SINR(dB)'] = [round(float(val), 8)]

    # --------- 放 Action Phase (最後) ---------
    if isinstance(action_phase, torch.Tensor):
        action_phase = action_phase.detach().cpu().numpy().flatten().tolist()
    elif isinstance(action_phase, np.ndarray):
        action_phase = action_phase.flatten().tolist()

    action_phase_str = json.dumps(action_phase)
    data["Action(Phase)"] = [action_phase_str]

    # --------- 儲存到 CSV ---------
    df_state = pd.DataFrame(data)

    # 確保資料夾存在
    folder_name = os.path.dirname(state_csv_file_path)  # 取得檔案路徑中的資料夾部分
    print(f"folder_name: {folder_name}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 使用 'a' 模式附加數據, 如果文件不存在則寫入標題
    file_exists = os.path.exists(state_csv_file_path)
    # print(f"csv_file_path: {state_csv_file_path}")
    df_state.to_csv(state_csv_file_path, mode='a', index=False, header=not file_exists)
    print(f"SINR and Datarate data BEFORE/AFTER adjustment have been saved to the {state_csv_file_path}")

    # --------- 儲存所有 parser.add_argument 設定 ---------
    # === 儲存 parser_config.json ===
    config_dict = vars(args).copy()
    config_dict.update(extra_config)

    # 把 tensor 轉成 list 或 float（防止 JSON dump 出錯）
    def tensor_to_serializable(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist() if obj.dim() > 0 else obj.item()
        elif isinstance(obj, dict):
            return {k: tensor_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_serializable(v) for v in obj]
        else:
            return obj

    config_dict = tensor_to_serializable(config_dict)

    json_dir = os.path.dirname(state_csv_file_path)
    json_path = os.path.join(json_dir, "parser_config.json")
    with open(json_path, "w") as f:
        json.dump(config_dict, f, indent=4)
    
    
    # parser_config_path = os.path.join(folder_name, 'parser_config.json')
    # try:
    #     def safe_convert(obj):
    #         if isinstance(obj, (np.ndarray, torch.Tensor)):
    #             return obj.tolist()
    #         elif isinstance(obj, (int, float, str, bool, list, dict)):
    #             return obj
    #         else:
    #             return str(obj)

    #     config_to_save = {k: safe_convert(v) for k, v in vars(args).items()}

    #     if extra_config:
    #         for k, v in extra_config.items():
    #             config_to_save[k] = safe_convert(v)

    #     with open(parser_config_path, 'w') as f:
    #         json.dump(config_to_save, f, indent=4)
    #     print(f"Parser + scenario settings saved to {parser_config_path}")

    # except Exception as e:
    #     print(f"Failed to save parser_config.json: {e}")

if __name__ == '__main__':
    
    init_display_settings()

    try:
        parser = argparse.ArgumentParser()                                      # 使用 argparse 定義和解析命令行參數
        parser.add_argument('--cpu_core', default=None, type=int, help='Specify the CPU core to use')
        parser.add_argument('--seed', default=None, type=int)                    # random seed, 128, None
        parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
        parser.add_argument('--use_cuda', default=True, type=bool)

        parser.add_argument('--fixed_ue', default=True, action='store_true', help='Use fixed UE positions')
        parser.add_argument('--neuron', default=64, type=int, help='Number of neurons in each layer')
        parser.add_argument('--max_episodes', default=1000, type=int)           # 800 timeslot, default:501, train 0.7 testint 0.3
        parser.add_argument('--episode_length', default=800, type=int)         # 400, temp_step
        parser.add_argument('--batch_size', default=128, type=int)              # batch size, 16 32 64 128
        
        # parser.add_argument('--lr', default=0.005, type=float)                   # learning rate: 0.1
        parser.add_argument('--lr_actor', default=5e-4, type=float)           # Actor 學習率
        parser.add_argument('--lr_critic', default=5e-4, type=float)           # Critic 學習率, 通常比 actor 大 10 倍
        '''
            lr_actor (Actor 學習率):
                越大 >> Actor 會更快更新策略, 但可能導致不穩定的策略變動 (容易來回震盪)
                越小 >> 策略變動會更平滑, 但學習可能變慢, 難以適應環境變化
            lr_critic (Critic 學習率):
                越大 >> Critic 可以更快學習 Q-value, 但可能導致 Q-value 過度樂觀, 影響 Actor 學習
                越小 >> Q-value 估計更準確, 但可能會學習太慢, 導致 Actor 訓練不夠有效
        '''

        # parser.add_argument('--alpha', default=0.3, type=float)                # entropy term coefficient: 0.1
        parser.add_argument('--gamma', default=0.2, type=float)                # Bellman Equation 中的 gamma, 是 discount factor
        '''
            如果 gamma 越大 (接近 1.0), 代表 RL 更關心未來的 reward:
                適合長期影響較大的問題, 例如每個 RIS phase 調整都會影響接下來許多時刻
                適用於 時序關聯性較強的環境 (如 navigation、inventory management)
            如果 gamma 越小 (接近 0.9), 代表 RL 更關心當前的 reward:
                適合你的場景, 因為 每個 timeslot 都重新調整 RIS phase, 並不影響未來時刻

            RIS 設定是每個 timeslot 都會重新調整 phase, 所以它的 phase 變動對後續的影響不會延續很久, 這意味著：
                若每個 timeslot 的 phase 對未來幾個 timeslot 有影響, 就要設高一點 (gamma = 0.98)
                但如果每個 timeslot 獨立運作, 則 gamma 設低 (gamma = 0.90 或 0.95) 可能更合理
        '''

        parser.add_argument('--tau', default=0.001, type=int)                   
        '''
            tau 越大 (如 tau = 0.01 或 0.05):
                目標網路更新得快, 變得和當前網路更接近
                優點: 學習變快, 但可能導致學習不穩定, 特別是在 Critic loss 變化很大時
                適合情境: 
                    環境變化快, 需要讓目標網路適應快速變動的策略
                    Critic loss 不會劇烈波動
            tau 越小 (如 tau = 0.001 或 0.005):
                目標網路更新得慢, 與當前網路保持較大差距
                優點: 學習更穩定, 減少梯度爆炸或發散的可能
                缺點: 學習速度變慢
                適合情境:
                    Critic loss 變化很大, 需要讓目標網路穩定學習
                    發現梯度爆炸或 loss 劇烈變化時
        '''
        parser.add_argument('--policy_delay', default=20, type=int)              # 5, 每間隔 policy_delay 次更新 Actor

        parser.add_argument('--exploration_rate', default=1.0, type=float)        # 1.0, 初始 100% 探索, 完全隨機選擇 phase, 幫助 RL 發現較好的動作
        parser.add_argument('--exploration_decay', default=0.995, type=float)     # 0.995, 每次訓練後探索率乘上這個值, 越高表示探索階段會更長
        parser.add_argument('--exploration_min', default= 0.1, type=float)       # 0.05, 最低探索率, 幾乎選擇最好的動作
        parser.add_argument('--ema_beta', default=0.2, type=float)                # 0.9, Exponential Moving Average (EMA) beta, 用來平滑 actor 和 critic 的 loss, 若越接近1表示記憶效果越長久(長期平均), 平滑程度高, 會以緩慢的速度逐漸追上新的reward; 越接近0表示更注重當前值(短期反應), 變動大, 會以較快的速度逐漸追上新的reward
        parser.add_argument('--gumbel_tau', default=1.0, type=float)        # Gumbel softmax temperature, 1.0

        parser.add_argument('--buffer_size', default=int(1e7), type=int)        # buffer_size與batch_size對應: 1e4(64), 1e5(128), 1e6(256)
        # parser.add_argument('--exploration_noise', default=0.1, type=float)     # exploration noise, 0.1
        
        # parser.add_argument('--mu', default=0.0, type=float)
        # parser.add_argument('--theta', default=0.15, type=float)
        # parser.add_argument('--sigma', default=0.2, type=float)
        # parser.add_argument('--noise_std', default=0.6, type=float)             # 探索的標準差
        # parser.add_argument('--noise_decay', default=0.999, type=float)         # 噪聲衰減率
        
        parser.add_argument("--bits", default=1,  type=int)                      # RIS phase shift bit count (2 bits (4), 4 bits (16), 6 bits (64), 8 bits (256))
        parser.add_argument("--group_size", default=(8,8), type=tuple)         # 設定 group size

        parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
        parser.add_argument("--save_data", default=True, type=bool)             # loss , reward
        parser.add_argument("--save_model", default=True, type=bool)            # model
        parser.add_argument("--load_model", default=False, type=bool)           # load model
        parser.add_argument("--load_model_name", default='20241002_113848_model.pth', type=str)       # load model
        # load model 指令: python main.py --load_model True --load_model_name ./model/20241002_113848_model.pth

        args = parser.parse_args()

        if args.cpu_core is not None:
            os.sched_setaffinity(0, {args.cpu_core})  # 綁定到指定 CPU 核心

        main(args)

    finally:
        # Clear the cache and perform garbage collection
        torch.cuda.empty_cache()
        gc.collect()

"""
Command:
    Train: python main.py --cpu_core 0
    Testing (load model): python main.py --load_model True --load_model_name ./model/20240921_222548_model.pth
"""