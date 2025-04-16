#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import csv
import time
import numpy as np
import matplotlib.pyplot as plt

class SaveCSV(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        # # 使用時間格式建立資料夾名稱
        # folder_name = './csv/' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # 確保資料夾存在, 不存在則創建
        os.makedirs(self.folder_name, exist_ok=True)
        self.file_index = 1
        self.filename = f'{self.folder_name}/RL_training_{self.file_index}.csv'
        self.create_new_file()

    def create_new_file(self):
        # 建立新的 CSV 檔案並寫入標題
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode(timeslot)', 'agent', 'step', 'temp_step', 'reward', 'critic_loss', 'actor_loss', 'action'])

    def check_file_size(self):
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 45 * 1024 * 1024:  # 45MB
            self.file_index += 1
            self.filename = f'{self.folder_name}/RL_training_{self.file_index}.csv'
            self.create_new_file()

        # # 更新檔案路徑
        # self.filename = self.folder_name + '/RL_training.csv'
        # with open(self.filename, 'w', newline='') as csvfile:  # 開啟輸出的 CSV 檔案
        #     writer = csv.writer(csvfile)  # 建立 CSV 檔寫入器
        #     writer.writerow(['Episode(timeslot)', 'agent', 'step', 'temp_step', 'reward', 'loss', 'action', 'Q_value'])

    def insert(self, epsi, agent, step, temp_step, reward, critic_loss, actor_loss, action):
        # print(self.filename)
        self.check_file_size()  # 檢查檔案大小
        with open(self.filename, 'a', newline='') as csvfile:  # 開啟輸出的 CSV 檔案
            writer = csv.writer(csvfile)  # 建立 CSV 檔寫入器
            writer.writerow([epsi, agent, step, temp_step, reward, critic_loss, actor_loss, action])

class SaveCSV_testing(object):
    def __init__(self, folder_name):
        self.folder_name = folder_name
        # # 使用時間格式建立資料夾名稱
        # folder_name = './csv/' + time.strftime("%Y%m%d_%H%M%S", time.localtime())
        # 確保資料夾存在, 不存在則創建
        os.makedirs(self.folder_name, exist_ok=True)
        self.file_index = 1
        self.filename = f'{self.folder_name}/RL_testing_{self.file_index}.csv'
        self.create_new_file()

    def create_new_file(self):
        # 建立新的 CSV 檔案並寫入標題
        with open(self.filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode(timeslot)', 'agent', 'reward', 'action'])

    def check_file_size(self):
        if os.path.exists(self.filename) and os.path.getsize(self.filename) > 45 * 1024 * 1024:  # 45MB
            self.file_index += 1
            self.filename = f'{self.folder_name}/RL_testing_{self.file_index}.csv'
            self.create_new_file()

        # # 更新檔案路徑
        # self.filename = self.folder_name + '/RL_testing.csv'
        # with open(self.filename, 'w', newline='') as csvfile:  # 開啟輸出的 CSV 檔案
        #     writer = csv.writer(csvfile)  # 建立 CSV 檔寫入器
        #     writer.writerow(['Episode(timeslot)', 'agent', 'reward', 'action'])

    def insert(self, epsi, agent, reward, action):
        #print(self.filename)
        self.check_file_size()  # 檢查檔案大小
        with open(self.filename, 'a', newline='') as csvfile:  # 開啟輸出的 CSV 檔案
            writer = csv.writer(csvfile)  # 建立 CSV 檔寫入器
            writer.writerow([epsi, agent, reward, action])


'''
class SaveChart(object):
    def __init__(self):
        self.x = []
        self.y = []
        plt.ion()
        self.figure, ax = plt.subplots(figsize=(8,6))
        self.chart, = ax.plot(self.x, self.y)

        plt.xlabel("Episode")
        plt.ylabel("Loss")

    def insert(self, epsi , loss):
        self.x.append(epsi)
        self.y.append(loss)
        
        self.chart.set_xdata(self.x)
        self.chart.set_ydata(self.y)
        print(self.x)
        print(self.y)
        self.figure.canvas.draw()
            
        self.figure.canvas.flush_events()
        #time.sleep(1)
'''

# class SaveChartCSV(object):
#   def __init__(self):
#     # 確保目錄存在，不存在則創建
#     os.makedirs('./csv', exist_ok=True)
#     # 更新檔案路徑
#     self.filename = './csv/Chart_' + time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '.csv'
#     with open(self.filename, 'w', newline='') as csvfile:        # 開啟輸出的 CSV 檔案
#       writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
#       writer.writerow(['step', 'reward', 'c_loss', 'a_loss'])

#   def insert(self, step , reward ,c_loss , a_loss):
#     #print(self.filename)
#     with open(self.filename, 'a', newline='') as csvfile:          # 開啟輸出的 CSV 檔案
#       writer=csv.writer(csvfile)                            # 建立 CSV 檔寫入器
#       writer.writerow([step, reward, c_loss , a_loss])
