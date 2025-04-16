#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import csv
import time
import numpy as np
import matplotlib.pyplot as plt

from utils import scenario_configs
from utils import scenario

# 從 utils.py 的 scenario_configs 函數中獲取與指定場景相關的各種配置參數。
wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode = scenario_configs(scenario)

num_agents = ris_size[0]*ris_size[1]
print('num_elements: ', num_agents)

class SaveQValue(object):
    def __init__(self, num_agents, folder_name):
        self.folder_name = folder_name
        # # 使用時間格式建立資料夾名稱
        # folder_name = './csv/' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # 確保資料夾存在, 不存在則創建
        os.makedirs(self.folder_name, exist_ok=True)

        # 更新檔案路徑
        self.filename = self.folder_name + 'Q-value.csv'
        with open(self.filename, 'w', newline='') as csvfile:        # 開啟輸出的 CSV 檔案
            writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
            writer.writerow(['Episode', 'step', 'element_no', 'Q-value' ])          # default: msg

    def insert(self, epsi, step, element_no, value):                        # default: msg
        #print(self.filename)
        with open(self.filename, 'a', newline='') as csvfile:          # 開啟輸出的 CSV 檔案
            writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
            writer.writerow([epsi, step, element_no, value])             # default: msg

class SaveAction(object):
    def __init__(self, num_agents, folder_name):
        self.folder_name = folder_name
        # # 使用時間格式建立資料夾名稱
        # folder_name = './csv/' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # 確保資料夾存在, 不存在則創建
        os.makedirs(self.folder_name, exist_ok=True)

        # 更新檔案路徑
        self.filename = self.folder_name + 'Action.csv'
        with open(self.filename, 'w', newline='') as csvfile:        # 開啟輸出的 CSV 檔案
            writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
            headers = ['Episode', 'step'] + [f'agent{i+1}' for i in range(num_agents)]
            writer.writerow(headers)
        
    def insert(self, epsi, step, agents):
        #print(self.filename)
        with open(self.filename, 'a', newline='') as csvfile:          # 開啟輸出的 CSV 檔案
            writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
            writer.writerow([epsi, step] + agents)

# for test
# num_agents = ris_size[0]*ris_size[1]             # 1024 OR 480
# save_action = SaveAction(num_agents)
# save_action.insert(1, 1, [0] * num_agents)  # 假設每個 agent 的動作值是 0