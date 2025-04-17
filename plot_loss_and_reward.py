import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import ScalarFormatter
import warnings
import numpy as np
import random
import glob
import torch

# 定義顏色、線條樣式和標記
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
linestyles = ['-', '--', ':', '-.', (0, (5, 10)), (0, (3, 1, 1, 1)), (0, (1, 2, 1, 2))]
# markers = ['o', 's', '*', '^', 'D', 'x', '+']

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

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_unique_colors(n):
    # 產生隨機顏色並確保它們不重複
    color_random = set()
    while len(color_random) < n:
        color_random.add((random.random(), random.random(), random.random()))
    return list(color_random)

def plot_all_timeslots_step(filtered_data, metric, ylabel, title, filename, save_path, min_timeslot, max_timeslot):

    init_display_settings()

    # 處理圖例放置於圖表外側(超過10個timeslots的話)
    if max_timeslot - min_timeslot + 1 > 10:
        plt.figure(figsize=(18, 6), dpi=300)
        
        lines = []  # 用來儲存線條對象
        labels = []  # 用來儲存標籤

        # 定義顏色循環
        colors_multi = generate_unique_colors(max_timeslot - min_timeslot + 1)
        
        for idx, timeslot in enumerate(range(min_timeslot, max_timeslot + 1)):
            subset = filtered_data[filtered_data['Episode(timeslot)'] == timeslot]
            color = colors_multi[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            # marker = markers[idx % len(markers)]  # 循環使用標記
            
            # # 每100筆資料標記一個點
            # x_data = subset['step']
            # y_data = subset[metric]
            # marker_every = max(1, len(x_data) // 100)
            # line, = plt.plot(
            #     x_data, y_data, 
            #     color=color, linestyle=linestyle, linewidth=1, alpha=0.5, 
            #     marker='o', markevery=marker_every
            # )
            line, = plt.plot(                     # linewidth是線條粗細, alpha是透明度, marker是標記形狀
                subset['step'], 
                subset[metric], 
                color=color, linestyle=linestyle, linewidth=1, alpha=0.5
            )      
            # line, = plt.plot(subset['step'], subset[metric], color=color, linestyle=linestyle, linewidth=1, marker=marker, alpha=0.5)     # linewidth是線條粗細, alpha是透明度, marker是標記形狀
            
            lines.append(line)
            labels.append(f'timeslot {timeslot}')

        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.title(title)

        n_cols = (len(lines) + 19) // 20    # 計算圖例需要的列數
        right_value = max(0.5, 0.8 - 0.01 * n_cols)  # 确保 `right` 不会过小
        left_value = min(0.2, right_value - 0.1)     # 确保 `left` 小于 `right`

        plt.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., ncol=n_cols)
        plt.subplots_adjust(right=right_value, left=left_value)  # 調整子圖以為圖例留出空間, 0.05
        # plt.subplots_adjust(right=0.8 - 0.01 * n_cols, left=0.07)  # 調整子圖以為圖例留出空間, 0.05
    else:
        plt.figure(figsize=(10, 6), dpi=300)

        for idx, timeslot in enumerate(range(min_timeslot, max_timeslot + 1)):
            subset = filtered_data[filtered_data['Episode(timeslot)'] == timeslot]
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            # marker = markers[idx % len(markers)]  # 循環使用標記
            
            # # 每100筆資料標記一個點
            # x_data = subset['step']
            # y_data = subset[metric]
            # marker_every = max(1, len(x_data) // 100)
            # line, = plt.plot(
            #     x_data, y_data, 
            #     label=f'timeslot {timeslot}',
            #     color=color, linestyle=linestyle, linewidth=1, alpha=0.5, 
            #     marker='o', markevery=marker_every
            # )
            plt.plot(                       # linewidth是線條粗細, alpha是透明度, marker是標記形狀
                subset['step'], subset[metric], 
                label=f'timeslot {timeslot}', 
                color=color, linestyle=linestyle, linewidth=1, alpha=0.5
            )  
            # plt.plot(subset['step'], subset[metric], label=f'timeslot {timeslot}', color=color, linestyle=linestyle, linewidth=1, marker=marker, alpha=0.5)  # linewidth是線條粗細, alpha是透明度, marker是標記形狀
        
        plt.xlabel('Step')
        plt.ylabel(ylabel)
        plt.title(title)
        
        plt.legend(loc='best')
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(style='plain', axis='y')
        ax.xaxis.set_major_formatter(ScalarFormatter())  # 設定 x 軸為整數顯示
        ax.ticklabel_format(style='plain', axis='x')     # 禁用 x 軸的科學記號

        plt.tight_layout()

    plt.savefig(os.path.join(save_path, filename))
    plt.show()

def plot_individual_timeslots_step(df, metric, ylabel, title_prefix, filename_prefix, save_path, min_timeslot, max_timeslot, dynamic_type):
    
    init_display_settings()

    for idx, timeslot in enumerate(range(min_timeslot, max_timeslot + 1)):
        filtered_df_indep = df[df['Episode(timeslot)'] == timeslot]
        if not filtered_df_indep.empty:
            plt.figure(figsize=(10, 6), dpi=300)
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            # marker = markers[idx % len(markers)]  # 循環使用標記
            
            # # 每100筆資料標記一個點
            # x_data = filtered_df_indep['step']
            # y_data = filtered_df_indep[metric]
            # marker_every = max(1, len(x_data) // 100)
            # line, = plt.plot(
            #     x_data, y_data, 
            #     label=f'timeslot {timeslot}',
            #     color=color, linestyle=linestyle, linewidth=1, alpha=0.5, 
            #     marker='o', markevery=marker_every
            # )
            plt.plot(                   # linewidth是線條粗細, alpha是透明度, marker是標記形狀
                filtered_df_indep['step'], filtered_df_indep[metric], 
                label=f'timeslot {timeslot}', 
                color=color, linestyle=linestyle, linewidth=1, alpha=0.5
            )
            # plt.plot(filtered_df_indep['step'], filtered_df_indep[metric], label=f'timeslot {timeslot}', color=color, linestyle=linestyle, linewidth=1, marker=marker, alpha=0.5)  # linewidth是線條粗細, alpha是透明度, marker是標記形狀

            plt.xlabel('Step')
            plt.ylabel(ylabel)
            plt.title(f'{title_prefix} for timeslot {timeslot}')
            plt.legend(loc='best')
            
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.ticklabel_format(style='plain', axis='y')
            ax.xaxis.set_major_formatter(ScalarFormatter())  # 設定 x 軸為整數顯示
            ax.ticklabel_format(style='plain', axis='x')     # 禁用 x 軸的科學記號

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'{filename_prefix}_{dynamic_type}_timeslot_{timeslot}.png'))
            plt.close()

def plot_loss_merge_timeslots_step(df, metric, ylabel, title, filename, save_path):
    
    init_display_settings()
    
    plt.figure(figsize=(10, 6), dpi=300)

    # 先排序數據, 按 timeslot 和 step 排序
    sorted_df = df.sort_values(by=['Episode(timeslot)', 'step'])

    # 重新計算 step, 讓它累計在一起
    cumulative_step = 0
    adjusted_steps = []

    unique_timeslots = sorted_df['Episode(timeslot)'].unique()
    
    for timeslot in unique_timeslots:
        subset = sorted_df[sorted_df['Episode(timeslot)'] == timeslot]
        steps = subset['step'] + cumulative_step  # 累計 step
        adjusted_steps.extend(steps)
        cumulative_step = steps.iloc[-1]  # 更新累計的步驟數
    
    # # 每100筆資料標記一個點
    # marker_every = max(1, len(adjusted_steps) // 10)
    # mark_indices = [0] + list(range(marker_every, len(adjusted_steps), marker_every))
    # # 繪製 metric 的連續變化圖
    # plt.plot(
    #     adjusted_steps, sorted_df[metric], 
    #     label=f'DQN(single agent centralized learning)',
    #     color='b', linestyle='-', linewidth=1, alpha=0.5,
    #     marker='o', markevery=mark_indices, markersize=5
    # )

    # plt.plot(
    #     adjusted_steps, sorted_df[metric], 
    #     label=f'DQN(single agent centralized learning)',
    #     color='b', linestyle='-', linewidth=1, alpha=0.7
    # )

    # 畫出 Critic Loss
    plt.plot(
        adjusted_steps, sorted_df['critic_loss'], 
        label='Critic Loss', 
        color='r', linestyle='-', linewidth=1.5, alpha=0.7
    )

    # 畫出 Actor Loss
    plt.plot(
        adjusted_steps, sorted_df['actor_loss'], 
        label='Actor Loss', 
        color='b', linestyle='--', linewidth=1.5, alpha=0.7
    )

    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='y')
    ax.xaxis.set_major_formatter(ScalarFormatter())  # 設定 x 軸為整數顯示
    ax.ticklabel_format(style='plain', axis='x')     # 禁用 x 軸的科學記號
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.show()

def plot_reward_merge_timeslots_step(df, metric, ylabel, title, filename, save_path):
    
    init_display_settings()
    
    plt.figure(figsize=(10, 6), dpi=300)

    # 先排序數據, 按 timeslot 和 step 排序
    sorted_df = df.sort_values(by=['Episode(timeslot)', 'step'])

    # 重新計算 step, 讓它累計在一起
    cumulative_step = 0
    adjusted_steps = []

    unique_timeslots = sorted_df['Episode(timeslot)'].unique()
    
    for timeslot in unique_timeslots:
        subset = sorted_df[sorted_df['Episode(timeslot)'] == timeslot]
        steps = subset['step'] + cumulative_step  # 累計 step
        adjusted_steps.extend(steps)
        cumulative_step = steps.iloc[-1]  # 更新累計的步驟數
    
    # # 每100筆資料標記一個點
    # marker_every = max(1, len(adjusted_steps) // 10)
    # mark_indices = [0] + list(range(marker_every, len(adjusted_steps), marker_every))
    # # 繪製 metric 的連續變化圖
    # plt.plot(
    #     adjusted_steps, sorted_df[metric], 
    #     label=f'DQN(single agent centralized learning)',
    #     color='b', linestyle='-', linewidth=1, alpha=0.5,
    #     marker='o', markevery=mark_indices, markersize=5
    # )

    plt.plot(
        adjusted_steps, sorted_df[metric], 
        label=f'DQN(single agent centralized learning)',
        color='b', linestyle='-', linewidth=1, alpha=0.7
    )

    plt.xlabel('Step')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax.ticklabel_format(style='plain', axis='y')
    ax.xaxis.set_major_formatter(ScalarFormatter())  # 設定 x 軸為整數顯示
    ax.ticklabel_format(style='plain', axis='x')     # 禁用 x 軸的科學記號
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, filename))
    plt.show()

def main():

    init_display_settings()

    # 動態部分, 時間戳與類型
    dynamic_time = '20250415_160929_seed_128'
    dynamic_type = 'training'  # training or testing

    # 動態生成 CSV 文件路徑
    csv_folder_path = f'./csv/{dynamic_time}/'

    # 使用 glob 模組讀取所有符合條件的 CSV 檔案
    csv_files = glob.glob(os.path.join(csv_folder_path, 'RL_training_*.csv'))

    # 確保儲存圖的資料夾存在
    save_path = f'./plot/{dynamic_time}/'
    ensure_directory_exists(save_path)

    # 讀取並合併所有 CSV 檔案, 排除標題行
    df_list = [pd.read_csv(file, header=0) for file in csv_files]  # header=0 只會讀取一次標題行, 然後自動排除重複的標題
    # print(f'df_list[0]: {df_list[0]}, df_list[1]: {df_list[1]}')
    df = pd.concat(df_list, ignore_index=True)
    # print(f'df.iloc[829439]: {df.iloc[829439]}')      # RL_training_1.csv 的最後一行
    # print(f'df.iloc[829440]: {df.iloc[829440]}')      # RL_training_2.csv 的第一行

    # 找出 timeslot 的最小與最大值
    min_timeslot = df['Episode(timeslot)'].min()
    max_timeslot = df['Episode(timeslot)'].max()

    # 過濾出指定範圍內的 timeslot 數據
    filtered_df = df[(df['Episode(timeslot)'] >= min_timeslot) & (df['Episode(timeslot)'] <= max_timeslot)]

    # 判斷是 training 還是 testing
    if dynamic_type == 'training':

        # # 所有timslots畫在同一張圖, 繪製 Loss, Reward 和 Q-value 圖形
        # plot_all_timeslots_step(filtered_df, 'loss', 'Loss', f'Loss over Steps for timeslot {min_timeslot} to {max_timeslot}', f'loss_{dynamic_type}.png', save_path, min_timeslot, max_timeslot)
        # plot_all_timeslots_step(filtered_df, 'reward', 'Reward', f'Reward over Steps for timeslot {min_timeslot} to {max_timeslot}', f'reward_{dynamic_type}.png', save_path, min_timeslot, max_timeslot)
        # # plot_all_timeslots_step(filtered_df, 'Q_value', 'Q-value', f'Q-value over Steps for timeslot {min_timeslot} to {max_timeslot}', f'Q-value_{dynamic_type}.png', save_path, min_timeslot, max_timeslot)

        # # 儲存每個 timeslot 的獨立 Loss, Reward 和 Q-value 圖形
        # plot_individual_timeslots_step(df, 'loss', 'Loss', 'Loss over Steps', 'loss', save_path, min_timeslot, max_timeslot, dynamic_type)
        # plot_individual_timeslots_step(df, 'reward', 'Reward', 'Reward over Steps', 'reward', save_path, min_timeslot, max_timeslot, dynamic_type)
        # # plot_individual_timeslots_step(df, 'Q_value', 'Q-value', 'Q-value over Steps', 'Q_value', save_path, min_timeslot, max_timeslot, dynamic_type)

        # 忽略 time slots, 繪製所有 step 的 Loss, Reward 和 Q-value
        plot_loss_merge_timeslots_step(df, 'loss', 'Loss', 'Training Loss across Steps', f'loss_{dynamic_type}_AcrossSteps.png', save_path)
        plot_reward_merge_timeslots_step(df, 'reward', 'Reward', 'Training Reward across Steps', f'reward_{dynamic_type}_AcrossSteps.png', save_path)
        # plot_merge_timeslots_step(df, 'Q_value', 'Q-value', 'Training Q-value across Steps', f'Q_value_{dynamic_type}_AcrossSteps.png', save_path)

    elif dynamic_type == 'testing':

        print('No ideas on how to present the test result figures!')

        # # 繪製 Reward 圖形
        # plot_metric(filtered_df, 'reward', 'Reward', f'Reward over Steps for timeslot {min_timeslot} to {max_timeslot}', f'reward_{dynamic_type}.png', save_path, min_timeslot, max_timeslot)

        # # 儲存每個 timeslot 的獨立 Reward 圖形
        # save_individual_plots(df, 'reward', 'Reward', 'Reward over Steps', 'reward', save_path, min_timeslot, max_timeslot, dynamic_type)

        

if __name__ == '__main__':
    main()