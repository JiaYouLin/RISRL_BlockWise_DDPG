#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
from RIS_RL.RISdate import V2Vstate ,Transmit_state

class V2VEnv(object):

    def __init__(self, MRB,timeslot,num_agent, **kwargs):
        self.config = kwargs['config']
        self.MRB = MRB
        self.timeslot = timeslot
        self.V2Vstate = 0
        self.num_agent = 0
        action_n = 16


    def step(self, actions ,agt):                                               # 根據動作計算下一狀態和獎勵
        #print(" >> ," ,agt)
        reward_n = []                                                           # 初始化獎勵、完成標誌和下一狀態
        done_n = []
        #action_n = actions
        #print(self.V2Vstate)
        next_state = self.V2Vstate.copy()

        # msg info，定義消息參數 msg_P、msg_D 和 msg_W。
        msg_P = [0.9999,0.9,0.9,0.99,0.9]
        msg_D = [100*1000,1000*1000,2500*1000,50*1000,2000*1000]
        msg_W = [2,1,1,1.5,1]

        # 計算選擇之CQI傳輸需要的RB數，並標註有傳輸的車輛
        action_RB = np.zeros( 5, dtype=np.int)                                  # 初始化每個消息所需的資源塊數量
        LeaderTrans = [[] for i in range(self.num_agent)]                       # 初始化每個代理的傳輸狀態
        for msg in range(5):                                                    # 根據動作計算每個消息所需的資源塊數量
            if actions[msg] == 0:        #不傳
                action_RB[msg] = 0
            else:
                action_RB[msg] = math.ceil(msg_D[msg]/RBDR_cal(actions[msg]))   #傳輸需要之RB數

        for agent in range(self.num_agent):
            LeaderTrans[agent] = 1
        #print("action_RB",action_RB)
        
        #並且計算傳輸時的SINR
        trans_state = Transmit_state(self.timeslot,self.MRB,LeaderTrans)        # 初始化傳輸狀態 trans_state
        PS = np.zeros((len(self.V2Vstate), 5))                                  # 紀錄有補足訊息的傳輸成功機率
        agent_reward = np.zeros(5)                                              # 初始化代理的reward

        sum_veh = 0 
        if np.sum(action_RB) <= self.config.max_uplink_RB:
            #print(self.V2Vstate)
        # 開始計算傳輸機率及計算reward
            for msg in range(5):
                utility = 0
                for veh in range(len(self.V2Vstate)):
                    if (self.V2Vstate[veh][3] !=1 ) and (self.V2Vstate[veh][4+msg] == 1) and (action_RB[msg] > 0):

                        leader_vehID = 0
                        for l in range(len(self.V2Vstate)):  #fine leader veh_ID
                            if self.V2Vstate[l][3] == 1:
                                leader_vehID = l
                                #print(leader_vehID)
                        #print(trans_state[veh][9],actions[msg],trans_state[leader_vehID][4+msg],trans_state[veh][4+msg],action_RB[msg])

                        # calculate_Probability(SINRv,NCQI,PL_prob,PM_prob,RB_num)
                        PS[veh][msg] = calculate_Probability(trans_state[veh][9],actions[msg],trans_state[leader_vehID][4+msg],trans_state[veh][4+msg],action_RB[msg])
                        #print(PS)
                        if PS[veh][msg] >= msg_P[msg] :
                            #print(agent)
                            next_state[veh][msg+4] = 0            # 補足訊息後state
                            utility += msg_D[msg]*msg_W[msg]        # 計算 utility
                            sum_veh +=1
                            agent_reward[msg] = round(utility/1048576,2)  # bits -> kbit
 
        else:
            for msg in range(5):
                agent_reward[msg] = - (int(action_RB[msg]))
                sum_veh += 0
        #print(np.sum(action_RB))           

        reward_n = agent_reward
        next_state[np.isnan(next_state)] = -10                              # 處理 nan 值
        next_state_temp = np.delete(next_state, [0,1,2,3], axis=1)          # 去除髒數據，刪除無用的狀態信息
        next_s = []
        for i in range(len(next_state_temp)):
            next_s.extend(next_state_temp[i])
        
        # 處理不同Agent(車隊)間維度不同問題(不足補0)，調整狀態維度，保證所有代理具有相同的維度
        list_size= 6 * 10        
        if len(next_s) < list_size:
            add_s = list_size-len(next_s)
            for _ in range(add_s):
                next_s.append(0)

        return next_s, reward_n, done_n ,sum_veh                            # 返回下一狀態、獎勵、完成標誌和總車輛數量


    def reset(self,agt):                                                    # 初始化環境狀態條件
        state_temp = V2Vstate(self.timeslot,self.MRB)                       # 獲取初始狀態
        self.V2Vstate = state_temp                                          # 更新狀態
        leader_num = 0                                                      # 計算領導者數量
        for leader in range(len(state_temp)):                               # 取得agent數
            leader_num += int(state_temp[leader][3])
        self.num_agent = leader_num
        state_temp = np.delete(state_temp, [0,1,2,3], axis=1)               # 去除髒數據，刪除無用狀態信息
        state_temp[np.isnan(state_temp)] = 0                                # 處理 nan 值

        #print(state_temp)
        state_temp = state_temp.tolist()                                    # 轉換為列表
        state=change_dimension(state_temp,self.num_agent,self.V2Vstate)     # 處理維度問題，調整狀態維度
        #print(state)

        # 過濾不屬於當前代理的狀態
        for z in range(len(state_temp)-1,-1,-1):
            if len(self.V2Vstate)>z and (self.V2Vstate[z][2] != agt+1).all():
                self.V2Vstate = np.delete(self.V2Vstate, z, axis=0)

        return state[agt]                                                   # 返回當前代理的狀態


# 為了要餵進去的神經網路，所以Agent數的維度與大小都要一樣，調整狀態維度
# 處理不同Agent(車隊)間維度不同問題
def change_dimension(state_temp,num_agent,complete_state):
    #處理 timeslot leader數不同問題(統一agent數)
    leader_sum = 0
    for num in range(len(complete_state)):                                  # 計算領導者數量
        if complete_state[num][3]:
            leader_sum += 1

    # 將一個Agent的obs轉成一維，初始化每個代理的狀態
    state = [[] for i in range(num_agent)]                                  # 整理每個Agent的state
    for i in range(len(complete_state)):
        for l in range(num_agent):
            if complete_state[i][2] == l+1 and l<=leader_sum:
                state[l].extend(state_temp[i])
        
    # 處理不同Agent(車隊)間維度不同問題(不足補0)，調整狀態維度，保證所有代理具有相同的維度
    list_size=0                                
    for s in range(num_agent):
        if len(state[s]) > list_size:
            list_size = len(state[s])
    for a in range(num_agent):
        if len(state[a]) < list_size:
            add_s = list_size-len(state[a])
            for _ in range(add_s):
                state[a].append(0)
    return state                                                            # 返回調整後的狀態


# 根據 NCQI 計算效率
def RBDR_cal (NCQI) :
    if  NCQI == 1:
        efficiency=0.2344                                                   # 根據 NCQI 值設置效率
    elif NCQI == 2:
        efficiency=0.3770
    elif NCQI == 3:
        efficiency=0.6016
    elif NCQI == 4:
        efficiency=0.8770
    elif NCQI == 5:
        efficiency=1.1758
    elif NCQI == 6:
        efficiency=1.4766
    elif NCQI == 7:
        efficiency=1.9141
    elif NCQI == 8:
        efficiency=2.4063
    elif NCQI == 9:
        efficiency=2.7305
    elif NCQI == 10:
        efficiency=3.3223
    elif NCQI == 11:
        efficiency=3.9023
    elif NCQI == 12:
        efficiency=4.5234
    elif NCQI == 13:
        efficiency=5.1152
    elif NCQI == 14:
        efficiency=5.1152
    elif NCQI == 15:
        efficiency=5.5547
    else: 
        efficiency=0
    
    return 12*14*efficiency*1000                                            # 因為timeslot=1ms 所以*1000

# 計算消息傳輸成功機率
def calculate_Probability(SINRv,NCQI,PL_prob,PM_prob,RB_num):
    SINRn = NCQItoSINR(NCQI)                                                # 將 NCQI 轉換為 SINRn
    # 一個RB傳送成功的機率
    RB_p = (1/2) * (1 + math.erf((SINRv - SINRn + 1.285)/math.sqrt(2)))     # 計算單個資源塊的傳輸成功機率

    msg_probability = PM_prob+(1-PM_prob)*PL_prob*math.pow(RB_p, RB_num)
    #print(msg_probability)
    return msg_probability                                                  # 計算消息的傳輸成功機率

# 根據 NCQI 計算 SINR
def NCQItoSINR (NCQI):
    if NCQI == 1:
        SINRn=-6.658                                                        # 根據 NCQI 值設置 SINR
    elif NCQI == 2:
        SINRn=-4.098
    elif NCQI == 3:
        SINRn=-1.798
    elif NCQI == 4:
        SINRn=0.399
    elif NCQI == 5:
        SINRn=2.424
    elif NCQI == 6:
        SINRn=4.489
    elif NCQI == 7:
        SINRn=6.367
    elif NCQI == 8:
        SINRn=8.456
    elif NCQI == 9:
        SINRn=10.266
    elif NCQI == 10:
        SINRn=12.218
    elif NCQI == 11:
        SINRn=14.122
    elif NCQI == 12:
        SINRn=15.849
    elif NCQI == 13:
        SINRn=17.786
    elif NCQI == 14:
        SINRn=19.809
    else: 
       SINRn=19.809
       
    return SINRn                                                            # 返回 SINR
