#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np
import random

from network import DQN
from save import SaveQValue

g_step = 0      # 設定全域變數 g_step 為 0。


class Vehicles:

    def __init__(self, s_dim, a_dim, num_agent, **kwargs):                          # 接受狀態維度 (s_dim)、動作維度 (a_dim)、代理數量 (num_agent) 和其他關鍵字參數 (kwargs)。
        self.eval_net, self.target_net = DQN(s_dim, a_dim), DQN(s_dim, a_dim)       # 建立神經網路: 評估網路和目標網路
        self.s_dim = s_dim                                                          # state可觀察狀態數
        self.a_dim = a_dim                                                          # action可執行動作數
        self.config = kwargs['config']                                              # 從關鍵字參數中提取配置
        if self.config.save_data:                                                   # 根據配置檢查是否需要保存數據
            pass
            # self.save_q = SaveQValue()
        self.num_agent = num_agent                                                  # 初始化agnet數量
        self.device = 'cuda' if self.config.use_cuda else 'cpu'                     # 決定使用GPU還是CPU
        self.replay_buffer = list()                                                 # 初始化回放緩衝區
        self.memory_counter = 0                                                     # 初始化記憶計數器
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.config.lr)    # 使用Adam優化器 (eval_net參數和learning rate)
        self.loss_func = nn.MSELoss()                                               # 使用均方差損失函數 (loss(xi, yi)=(xi-yi)^2)
        self.learn_step_counter = 0                                                 # target，初始化學習步驟計數器
        self.Q_value_1 , self.Q_value_2 , self.Q_value_3 ,self.Q_value_4 ,self.Q_value_5 = 0,0,0,0,0        # 初始化Q值

        self.loss = 0                                                               # 初始化損失

        if self.config.use_cuda:                                                    # use GPU(cuda)，如果使用CUDA，則將網路移動到GPU上
            self.eval_net.cuda()
            self.target_net.cuda()
   
        if self.config.load_model:                                                  # load model，如果需要載入模型，則從檔案中載入模型和優化器的狀態
            checkpoint = torch.load(self.config.load_model_name)
            self.eval_net.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.loss = checkpoint['loss']
            print("Agent load model success!")

    def memory(self,agt, s, a, r, s_, done):                                        # replay buffer，用於儲存回放緩衝區的經驗
        self.replay_buffer.append((agt ,s, a, r, s_, done))                         # 將經驗添加到回放緩衝區

        if self.memory_counter >= self.config.memory_length:                        # 如果記憶體計數器超過最大記憶長度，則從回放緩衝區中移除最舊的經驗
            self.replay_buffer.pop(0)                                               # 移除列表中的一個元素（預設最後一個元素）
            self.memory_counter += 1                                                # 增加記憶體計數器

    def choose_action(self, s, step,noisy=True):                                    # 基於epsilon-greedy策略，根據狀態s和步驟step選擇動作
        if self.config.use_cuda:                                                    # 如果使用CUDA，將狀態s轉換為GPU張量
            s = torch.unsqueeze(torch.cuda.FloatTensor(s), 0)
        else:
            s = torch.unsqueeze(torch.FloatTensor(s), 0)

        actions_value = np.zeros((5, self.a_dim), dtype=np.float)                   # 產生array來存放不同Agent的所觀察到的state做出的action
        actions = np.zeros( 5, dtype=np.int)                                        # 產生array來存放不同Agent的所觀察到的state做出的action value

        if np.random.uniform() < self.config.epsilon_start + step / (self.config.episode_length-800):       # 根據epsilon-greedy方法選擇action，如果滿足條件，則使用eval_net計算Q值並選擇最大Q值對應的動作
            for msg in range(5):
                actions_value[msg] = self.eval_net.forward(s)[msg][0].detach().cpu().numpy()
            # 選出最大Q-value的action
            for msg in range(5):
                action_idx = np.argmax(actions_value[msg])
                actions[msg] = action_idx        

        else:           # 否則，隨機選擇動作
            print("\n\n **  random now ! espilon : " , (self.config.epsilon_start + step / self.config.episode_length))
            for msg in range(5):
                actions[msg]=random.randint(0, 15)        

        print(actions)        
        global g_step           # 更新全域變數g_step
        g_step = step

        return actions          # 返回選擇的動作


    def get_batches(self):          # 取得批次資料的函式，隨機抽取回放緩衝區中的經驗，用於訓練
        experiences = random.sample(self.replay_buffer, self.config.batch_size)     # 隨機抽取一批經驗
        state_batches = np.array([_[1] for _ in experiences])                       # 提取狀態
        action_batches = np.array([_[2] for _ in experiences])                      # 提取動作
        reward_batches = np.array([_[3] for _ in experiences])                      # 提取獎勵
        next_state_batches = np.array([_[4] for _ in experiences])                  # 提取下一個狀態
        done_batches = np.array([_[5] for _ in experiences])                        # 提取完成標誌
        return state_batches, action_batches, reward_batches, next_state_batches, done_batches      # 返回這些批次資料


    def train(self):                # 訓練函式，更新目標網路，取出回放緩衝區中的資料，計算Q值，更新網路參數
        if self.learn_step_counter % self.config.target_per_update == 0:            # 每target_per_update步更新一次目標網路
            self.target_net.load_state_dict(self.eval_net.state_dict())             # 將評估網路的參數给目標網路
        self.learn_step_counter += 1                                                # learn step +1

        # 取出 replay buffer data
        state_batches, action_batches, reward_batches, next_state_batches, done_batches = self.get_batches()        # 取得批次資料

        # 將批次資料轉換為 CUDA 張量
        state_batches = torch.FloatTensor(state_batches).cuda()
        action_batches = torch.LongTensor(action_batches).cuda()
        reward_batches = torch.FloatTensor(reward_batches).cuda()
        next_state_batches = torch.FloatTensor(next_state_batches).cuda()
        done_batches = torch.FloatTensor((done_batches == False) * 1).cuda()

        # 取出batche size大小的估計值和目標值，利用損失函數和優化器估計網路更新
        # eval_net 透過state輸出估計網路每個對應的action值，gather表示對應每個state的action value(Q值)
        # 因為我們要針對5種不同訊息,因此會output 5個 Q-value
        eval_net_temp = self.eval_net(state_batches)                                        # 通過評估網路前向傳播計算Q值

        # 從評估網路中提取Q值
        q_eval_msg1 = eval_net_temp[0].gather(1,action_batches[: ,0].unsqueeze(1))          # q_eval
        q_eval_msg2 = eval_net_temp[1].gather(1,action_batches[: ,1].unsqueeze(1))          # q_eval
        q_eval_msg3 = eval_net_temp[2].gather(1,action_batches[: ,2].unsqueeze(1))          # q_eval
        q_eval_msg4 = eval_net_temp[3].gather(1,action_batches[: ,3].unsqueeze(1))          # q_eval
        q_eval_msg5 = eval_net_temp[4].gather(1,action_batches[: ,4].unsqueeze(1))          # q_eval

        # q_next表示透過目標網路输出batche size大小的下個state對應的action value，從目標網路中計算下一狀態的Q值
        # 使用detach()是因為q_next不進行反向傳播
        q_next_msg1 = self.target_net(next_state_batches)[0].detach()                       # q_next
        q_next_msg2 = self.target_net(next_state_batches)[1].detach()                       # q_next
        q_next_msg3 = self.target_net(next_state_batches)[2].detach()                       # q_next
        q_next_msg4 = self.target_net(next_state_batches)[3].detach()                       # q_next
        q_next_msg5 = self.target_net(next_state_batches)[4].detach()                       # q_next
        #print("q_next")
        #print(q_next_msg1,"\n",q_next_msg2,"\n",q_next_msg3,"\n",q_next_msg4,"\n",q_next_msg5)

        # 透過公式得到目標值，計算目標Q值
        # max(1)[0]表示選出每行的最大值，view()表示把前面得到的一维(BATCH_SIZE, 1)的形狀
        q_target_msg1 = reward_batches[:, 0].view(self.config.batch_size, 1)  + self.config.gamma * q_next_msg1.max(1)[0].view(self.config.batch_size, 1)
        q_target_msg2 = reward_batches[:, 1].view(self.config.batch_size, 1)  + self.config.gamma * q_next_msg2.max(1)[0].view(self.config.batch_size, 1)
        q_target_msg3 = reward_batches[:, 2].view(self.config.batch_size, 1)  + self.config.gamma * q_next_msg3.max(1)[0].view(self.config.batch_size, 1)
        q_target_msg4 = reward_batches[:, 3].view(self.config.batch_size, 1)  + self.config.gamma * q_next_msg4.max(1)[0].view(self.config.batch_size, 1)
        q_target_msg5 = reward_batches[:, 4].view(self.config.batch_size, 1)  + self.config.gamma * q_next_msg5.max(1)[0].view(self.config.batch_size, 1)

        # 計算loss，輸入評估值跟目標值，使用均方損失函數
        loss_msg1 = self.loss_func(q_eval_msg1, q_target_msg1)
        loss_msg2 = self.loss_func(q_eval_msg2, q_target_msg2)
        loss_msg3 = self.loss_func(q_eval_msg3, q_target_msg3)
        loss_msg4 = self.loss_func(q_eval_msg4, q_target_msg4)
        loss_msg5 = self.loss_func(q_eval_msg5, q_target_msg5)
        print(loss_msg1,"\n",loss_msg2,"\n",loss_msg3,"\n",loss_msg4,"\n",loss_msg5)
        loss = loss_msg1 + loss_msg2 + loss_msg3 + loss_msg4 + loss_msg5

        self.optimizer.zero_grad()                                                  # 清空上一步的梯度

        loss.backward()                                                             # 反向傳播

        self.optimizer.step()                                                       # 更新model參數

        self.loss = loss.item()
        
        # 更新Q值
        self.Q_value_1 = int(torch.sum(q_eval_msg1)) / self.config.batch_size
        self.Q_value_2 = int(torch.sum(q_eval_msg2)) / self.config.batch_size
        self.Q_value_3 = int(torch.sum(q_eval_msg3)) / self.config.batch_size
        self.Q_value_4 = int(torch.sum(q_eval_msg4)) / self.config.batch_size
        self.Q_value_5 = int(torch.sum(q_eval_msg5)) / self.config.batch_size


    def getLoss(self):                                                              # 用來獲取當前損失值
        return self.loss

    def getQvalue(self):                                                            # 用來獲取當前Q值
        return self.Q_value_1, self.Q_value_2, self.Q_value_3, self.Q_value_4, self.Q_value_5

    def getCheckpoint(self):                                                        # 用來獲取模型檢查點
        return self.eval_net, self.optimizer, self.loss