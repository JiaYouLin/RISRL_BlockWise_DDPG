import torch
import os
from channel import Channel, Beamformer
from utils import scenario_configs
from math import pi, log10
import math
import einops
 
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Union
from utils import gpu, scenario
import pandas as pd

import itertools

# 主要用於設置和運行一個模擬環境
if __name__ == '__main__':        
    # torch.set_printoptions(precision=12, threshold=10_000)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))                                        # 將當前工作目錄更改為腳本所在的目錄
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # scenario = 'general RIS 32x32_obstacles rectangle 3'

    print('=====================Parameters=====================')
    print(f'device: {device}')
    print(f'scenario: {scenario}')
    print('====================================================\n')

    # 從 utils.py 的 scenario_configs 函數中獲取與指定場景相關的各種配置參數
    wavelength, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode = scenario_configs(scenario)

    channel = Channel(
        wavelength=wavelength,
        ris_center=ris_center,
        ris_norm=ris_norm,
        ris_size=ris_size,
        area=area,
        BS_pos=BS_pos,
        M=M,
        K=K,
        device=device,
        MU_dist=MU_mode
    )
    beamformer = Beamformer(device=device)

    channel.config_scenario(
        centers_=centers,
        obstacles=obstacles,
        std_radius=std,
        scenario_name=scenario
    )
    channel.create()
    
    # ======================畫 test 場景圖=========================
    print('====================================================')
    # 確定存儲結果的目錄, 如果不存在, 則創建該目錄
    dir = os.path.abspath(f'./Scenario/test/{scenario}')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # 繪製並保存場景的阻塞條件和概覽圖
    channel.plot_block_cond(dir=dir)  # Plot block condition
    channel.show(dir=dir)  # plot the scenariot()

    n_env = 1                   # 1000 個模擬 element phase, 相當於 1000 個 timeslots                                                                                # number of parallel environments
    h_ris = ris_size[0]
    w_ris = ris_size[1]

    Z = torch.ones((n_env, h_ris*w_ris), device=device)
    print('==============Test 7 (Try all phase combinations by bits)==============')    
    # RIS phase shift bit 數 (2 bits(4)、4 bits(16)、6 bits(64)、8 bits(256))
    bits = 2
    
    # 定義相位選擇列表, 生成在 [0, 2pi] 區間內均勻分佈的 2^n 個相位值
    ideal_phases_full = np.linspace(0, 2*np.pi, 2**bits)

    # 將 numpy array 轉換為列表，以便分段處理
    phases = ideal_phases_full.tolist()
    # 顯示第一段（前 64 個值）
    # print(f'phases[:64]: {phases[:64]}, len: {len(phases)}\n')

    # 獲取並計算該場景中 RIS 實際的 element 數
    required_phase_count = ris_size[0] * ris_size[1]
    # print(f'required_phase_count: {required_phase_count}\n')

    # 生成所有 RIS element 相位的排列組合
    all_combinations = itertools.product(phases, repeat=required_phase_count)
    # print(f'all_combinations: {all_combinations}\n')

    # 初始化一個 DataFrame 來保存每個組合的結果
    results = pd.DataFrame()
    M = channel.M
    N = channel.N
    K = channel.K
    
    T = 100
    n_itr = T
    h_Bk = torch.view_as_complex(torch.zeros(n_itr, K, M, 2, device = device))
    H_BS = torch.view_as_complex(torch.zeros(n_itr, N, M, 2, device = device))
    H_Sk = torch.view_as_complex(torch.zeros(n_itr, K, N, 2, device = device))
    AWGN_norm = torch.view_as_complex(torch.zeros(n_itr, K, 1, 2, device =  device))
    
    for i in range(n_itr):
        h_Bk[i], H_BS[i], H_Sk[i], AWGN_norm[i] = channel.genCoef(2, 4, 10, time_corrcoef=0)
        # AWGN_norm[i] = 1/channel.mean_SNR
        # print(torch.abs(channel.AWGN_norm)**2 ,AWGN_norm[i])

    # # Beamform direction
    # D_BS = torch.mean(channel.D_BS, dim=1)       # (N)
    # D_Sk = channel.D_Sk                          # (K, N)
    # D = D_BS.unsqueeze(dim=0) + D_Sk             # (K, N)
    # D_Bk = torch.mean(channel.D_Bk, dim=1)       # (K)
                
    # wavelength = channel.wavelength
    # phase = 2*pi*(D-einops.repeat(D_Bk, 'k -> k n', n=D.size(dim=1)))/wavelength
    # Theta_dir = torch.polar(
    #     torch.ones_like(D, dtype=torch.float, device=device),
    #     phase,
    # )

    # 遍歷所有排列組合並計算 SINR 和 datarate
    for combo in tqdm(all_combinations, total=len(phases)**required_phase_count):

        # 將當前排列組合轉換為 tensor 並傳輸到 CUDA
        Theta_alldir_value = torch.tensor(combo, dtype=torch.float32).reshape(1, -1).to(device)
        print(f'Theta_alldir_value: {Theta_alldir_value}\n')

        Theta_all_indices = torch.polar(torch.ones_like(Theta_alldir_value, dtype=torch.float32), Theta_alldir_value).to(device)
        # print(f'Theta_fixed_indices: {Theta_fixed_indices}')

        Z_Theta = Z * Theta_all_indices
        SINR = 0
        for t in range(T):
            # training
            H = channel.calcJointCoef(h_Bk[t], Z_Theta, H_BS[t], H_Sk[t])
            W = beamformer.MRT(H)
            SINR += channel.SINR(H, W, AWGN_norm[i])                      # linear
            # SINR += channel.SINR(H, W, AWGN_norm[0])                      # linear
        SINR /= T
        SINR_db = 10*torch.log10(SINR).cpu().numpy()
        # print(f'Theta_testdir SINR(dB): {SINR_db}\n')
        # print(f'Theta_testdir SINR(dB) shape: {SINR_db.shape}\n')

        # PRINT: 將 sinr_db 展平成一維，然後格式化
        formatted_sinr_db = [f'{x:.10f}' for x in SINR_db.flatten()]
        print(f'Theta_testdir SINR(dB): {formatted_sinr_db}\n')

        datarate = channel.Calculate_datarate(SINR).cpu().numpy()
        print(f"datarate: {datarate}, shape: {datarate.shape} \n")
        
        # similarity = torch.abs(torch.sum(Theta_alldir_value * torch.conj(Theta_dir)))

        # 將 SINR 和 datarate 保存到 DataFrame
        row = pd.Series(list(combo) + list(SINR_db.flatten()) + list(datarate.flatten()))# + list(similarity.cpu().numpy().flatten()))
        results = pd.concat([results, row.to_frame().T], ignore_index=True)
    
    # 確保目錄存在
    csv_dir = f'./generate_RIS/all_phase_combinations/{scenario}'
    os.makedirs(csv_dir, exist_ok=True)
    # 將結果保存為 CSV 文件
    csv_path_all = os.path.join(csv_dir, 'all_phase_sinr_datarate_combinations.csv')
    results.to_csv(csv_path_all, index=True, header=True)  # 儲存整個 DataFrame

    print(f'All_phase_combinations saved to {csv_path_all}.')

    # #======================test 8 (random phase shift)======================
    # print('==============Test 8 (random phase shift)==============')
    # Theta_random = torch.rand_like(Z, device=device) * 2*pi
    # # print(f'Theta_random: {Theta_random}')

    # # PRINT: 將 Theta_random 展平成一維，然後格式化
    # formatted_Theta_random = [f'{x:.10f}' for x in Theta_random.flatten()]
    # # 只顯示前 10 個數據，後面加上省略號
    # print(f'Theta_random: {formatted_Theta_random[:60]}...\n')

    # # 定義離散相位狀態(以弧度為單位)
    # discrete_phases = torch.tensor([0, pi/2, pi, 3*pi/2], device=device)
    # # print(f'discrete_phases: {discrete_phases}\n')

    # # 建立與 Theta_random 相同形狀的張量來存儲離散相位
    # Theta_random_value = torch.zeros_like(Theta_random)

    # # 將連續相位轉換為離散相位
    # for i in range(Theta_random.size(0)):  # 遍歷每個環境
    #     for j in range(Theta_random.size(1)):  # 遍歷每個 RIS 元素
    #         # 計算到每個離散相位的距離
    #         distances = torch.abs(Theta_random[i, j] - discrete_phases)
    #         # 找到距離最近的離散相位的索引
    #         min_index = torch.argmin(distances)
    #         # 將最接近的離散相位分配給 Theta_random_value
    #         Theta_random_value[i, j] = discrete_phases[min_index]
    # # print(f'Theta_random_value: {Theta_random_value}\n')
    # # print(f'Theta_random_value shape: {Theta_random_value.shape}\n')

    # # PRINT: 將 Theta_random_value 展平成一維，然後格式化
    # formatted_Theta_random_value = [f'{x:.10f}' for x in Theta_random_value.flatten()]
    # # 只顯示前 10 個數據，後面加上省略號
    # print(f'formatted_Theta_random_value: {formatted_Theta_random_value[:60]}...\n')

    # # 使用離散相位生成極座標形式的 Theta_random_indices
    # Theta_random_indices = torch.polar(torch.ones_like(Z, dtype=torch.float32), Theta_random_value).to(device)
    # # print(f'Theta_random_indices: {Theta_random_indices}\n')

    # Z_Theta = Z * Theta_random_indices
    # H, AWGN_norm = channel.coef(Z_Theta, batch_size=None, progress=False)
    # W = beamformer.MRT(H)

    # # 計算 SINR
    # SINR = channel.SINR(H, W, AWGN_norm)                      # linear
    # SINR_db = 10*torch.log10(SINR).cpu().numpy()  
    # # print(f'Theta_random SINR(dB): {SINR_db}')
    # # print(f'Theta_random SINR(dB) shape: {SINR_db.shape}\n')

    # # PRINT: 將 sinr_db 展平成一維，然後格式化
    # formatted_sinr_db = [f'{x:.10f}' for x in SINR_db.flatten()]
    # print(f'Theta_random SINR(dB): {formatted_sinr_db}\n')

    # #=======================================================================

    # # print(f'Z.shape: {Z.shape}')        #([1, 1024])
    # # print(f'phase.shape: {Theta_random_value.shape}')        # ([1, 1024])
    # # print(f'Theta SINR(dB) shape: {SINR_db.shape}')              # ([n_env, 1024])
    
    # # 為符合皆為 tensor 型態, 將 SINR 轉為 tensor
    # SINR_db_tensor = torch.from_numpy(SINR_db).to(Theta_random_value.device)
    # # print(f'SINR_db_tensor: {SINR_db_tensor} \nshape: {SINR_db_tensor.shape}\n')

    # # 沿著 dim=1 拼接 phase 和 SINR
    # Phase_state = torch.cat((Theta_random_value, SINR_db_tensor), dim=1)
    # # print(f'Phase_state: {Phase_state}')
    # # print(f'Phase_state shape: {Phase_state.shape}\n')         # ([1, 1025])
    # """
    # 那我這邊應該是模擬1000次, 看是要先存起來還是每次呼叫, 但要有一個計數的, 這樣才知道丟了多少timeslots進去
    # 應該是每個element要是一筆資料, 有1024個element所以會有1024筆資料, 每一筆資料後面會再加上1個sinr

    # 1000 timeslot是再外層的資料夾, 所以是在timeslot 1時的這個csv會有250車
    # 因為尚桓的是每一台車是一筆資料, 有250台車所以會有250筆資料, 每一筆資料後面會再加上1個sinr
    # """

    # # ======================將每個 timeslot 的 Phase 與 UE 儲存為 CSV=========================
    # # =========將 Phase_state 依照 timeslot 分別保存為 CSV 文件=========
    # # 確保目錄存在
    # csv_dir = f'./generate_RIS/{scenario}'
    # os.makedirs(csv_dir, exist_ok=True)
    # Phase_state_np = Phase_state.cpu().numpy()  # 將張量轉換為 NumPy 陣列, 並移到 CPU 上以便於轉換
    
    # num_elements = h_ris * w_ris
    # num_ues = SINR_db_tensor.shape[1]
    # column_names = [f'element_{i+1}' for i in range(num_elements)] + [f'ue_{i+1}' for i in range(num_ues)]
    
    # # 指定行索引名稱
    # index_names = [f'timeslot_{i+1}' for i in range(n_env)]

    # # 將每個 row 存為獨立的 CSV 文件, 即每個 timeslot 獨立儲存於各 CSV
    # for i in range(Phase_state_np.shape[0]):
    #     row_data = Phase_state_np[i, :]

    #     df = pd.DataFrame([row_data])
    #     csv_path_emp = os.path.join(csv_dir, f'RIS_element_{h_ris*w_ris}')
    #     csv_path = f'{csv_path_emp}_Phase_timeslot{i+1}.csv'
    #     df.to_csv(csv_path, index=False, header=False)  # 儲存單獨的 CSV 文件, 純數值
    #     print(f'Saved timeslot {i+1} to '{csv_path}'.')

    #     df_marked = pd.DataFrame([row_data], columns=column_names)
    #     csv_path_marked_emp = os.path.join(csv_dir, f'RIS_element_{h_ris*w_ris}')
    #     csv_path_marked = f'{csv_path_marked_emp}_Phase_timeslot{i+1}_marked.csv'
    #     df_marked.to_csv(csv_path_marked, index=True, header=True)  # 儲存單獨的 CSV 文件, 有欄位名稱
    #     print(f'Saved timeslot {i+1} to '{csv_path_marked}'.')

    # # =========將所有 timeslot 的資料全部儲存於一個 CSV=========
    # all_data = []

    # for i in range(Phase_state_np.shape[0]):
    #     row_data = Phase_state_np[i, :]
    #     all_data.append(row_data)

    # # 創建 DataFrame 並指定列名
    # df_all = pd.DataFrame(all_data, columns=column_names)
    # df_all.index = index_names  # 設定行索引名稱

    # # 將 DataFrame 儲存為 CSV 文件
    # csv_path_all = os.path.join(csv_dir, 'RIS_element_Phase_all_timeslots_marked.csv')
    # df_all.to_csv(csv_path_all, index=True, header=True)  # 儲存整個 DataFrame

    # print(f'RIS_element_Phase for all timeslots saved to '{csv_path_all}'.')

    # # ======================畫場景圖=========================
    # print('====================================================')
    # # 確定存儲結果的目錄, 如果不存在, 則創建該目錄
    # dir = os.path.abspath(f'./Scenario/{scenario}')
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # # 繪製並保存場景的阻塞條件和概覽圖
    # channel.plot_block_cond(dir=dir)  # Plot block condition
    # channel.show(dir=dir)  # plot the scenario


