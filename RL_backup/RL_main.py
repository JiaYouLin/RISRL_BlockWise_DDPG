#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build env.py, add basic components, and visualize it.
"""
from env import V2VEnv

import argparse, datetime ,time
import numpy as np

from agents import Vehicles
import network 
import time

from output import SaveCSV , SaveCSV_testing
#from save import SaveAction


def main(args):

    #config param
    kwargs = dict()                                                         # kwargs：存儲配置參數
    kwargs['config'] = args


    state_dim = 6                                                           # state維度
    action_dim = 16                                                         # 有幾個action可以做，尚桓因為CQI
    num_agent = 27                                                          # agent數

    vehicle_model = Vehicles(state_dim, action_dim, num_agent, **kwargs)    # initial env

    episode = 2                                                             # 初始化 episode
    total_step = 0                                                          # 初始化 total_step
    if args.save_data:                                                      # Save data
        save_csv = SaveCSV()
        save_csv_testing = SaveCSV_testing()
        #save_action = SaveAction()


    while episode < args.max_episodes:
        env = V2VEnv(args.MRB,episode,num_agent,**kwargs)                   # 初始化環境 env
        # set env
        #state = env.reset()
        #print(state)
        step = 0                                                            # 初始化步驟計數器 step
        accum_reward = []                                                   # 初始化累積獎勵 accum_reward
        done = []
        pass_agt = 0

        for agt in range(num_agent):                                        # 對於每個代理 agt
            temp_step = 0
            state = env.reset(agt)                                          # 重置環境，獲取初始狀態
            while temp_step < args.episode_length:
                
                # 在每個 episode 中執行動作，更新狀態和獎勵
                action = vehicle_model.choose_action(state,step)
                next_state, reward, done,sum_veh = env.step(action,agt)
                step += 1
                total_step += 1
                print(reward)
                #exit()
                print("step = %d , sum_veh = %d" % (step,sum_veh))
                # print(reward)
                # print(np.sum(reward))

                accum_reward= np.sum(reward)
                done.append(False) 
                vehicle_model.memory(agt,state, action, reward, next_state, done)       # save in memory buffer

                # start training，如果回放緩衝區的長度大於批次大小，則進行訓練
                if len(vehicle_model.replay_buffer) >= args.batch_size:
                    vehicle_model.train()

                # 獲取當前的損失和 Q 值，並印出結果
                loss = vehicle_model.getLoss()       
                Q_value_1, Q_value_2, Q_value_3, Q_value_4, Q_value_5 = vehicle_model.getQvalue()  
                print("[Timeslot %05d ,Agent %05d] reward_vehicle %3.1f vehicle_loss %3.1f  \n\n" % \
                    (episode ,agt+1, np.sum(accum_reward).item(), loss))

                # 如果loss開始超過某個範圍 代表訓練已經過度，因此換下個episode
                if loss > 80 and step > 1500 and temp_step > 100:       
                    print(temp_step , step )
                    print("Loss too big! Pass.")
                    break
                
                # 如果需要保存數據，則將結果插入 CSV
                if args.save_data and (step % 10 == 0):
                    save_csv.insert(episode, agt+1 ,step,temp_step+1 , np.sum(accum_reward).item(), sum_veh ,loss ,action, Q_value_1, Q_value_2, Q_value_3, Q_value_4, Q_value_5 )
                    
                # 如果 episode 完成，則重置模型
                if True in done:
                    vehicle_model.reset()
                    break

                temp_step += 1

        episode += 1

        ## Save Model
        if args.save_model:
            print("\n\nModel Save : " + args.log_dir)
            ck_model,ck_optimizer,ck_loss = vehicle_model.getCheckpoint()       
            rl.save_model(ck_model,ck_optimizer,ck_loss,args.log_dir)
            #exit()
        #break

    # testing data
    print("************************\n\n")
    print("\n\n\n ========== Start Testing ========== ")
    for test_data in range(501, 626, 1):                                    # 對於測試數據範圍內的每個數據
        # 初始化環境，對於每個代理，重置環境，選擇動作，更新狀態和獎勵
        reward,sum_veh = 0,0
        env = V2VEnv(args.MRB,test_data,num_agent,**kwargs)
        for agt in range(num_agent):
            state = env.reset(agt)
            action = vehicle_model.choose_action(state,args.episode_length)
            next_state, temp_reward, done,temp_sum_veh = env.step(action,agt)
            reward = np.sum(temp_reward)
            sum_veh = temp_sum_veh

            # 印出結果並將結果插入 CSV
            print("[Timeslot %05d ,Agent %05d] reward_vehicle %3.1f sum_veh %3.1f  \n\n" % \
                (test_data,agt+1, reward, temp_sum_veh))
            save_csv_testing.insert(test_data, agt+1 ,reward, sum_veh,action )
                    

        #print("\n\n >> [Testing_timeslot %05d] reward_vehicle %3.1f vehicle_loss %3.1f  \n\n" % \
        #        (test_data, reward, sum_veh))            
        print("\n\n************************")

    #exit()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()                                      # 使用 argparse 定義和解析命令行參數
    ########## Env setting ##########
    parser.add_argument('--MRB', default=20, type=int)                      # Max uplink RB
    parser.add_argument('--max_uplink_RB', default=30, type=int)

    ########## Env setting ##########
    parser.add_argument('--max_episodes', default=501, type=int)            # 1000 timeslot
    parser.add_argument('--episode_length', default=5000, type=int) 
    parser.add_argument('--memory_length', default=int(2000), type=int)
    parser.add_argument("--target_per_update", default=100 ,type=int)       # 更新頻率
    parser.add_argument('--gamma', default=0.1, type=float)                 # discount factor
    parser.add_argument('--use_cuda', default=True, type=bool)
    parser.add_argument('--lr', default=0.01, type=float)                   # learning rate
    parser.add_argument('--batch_size', default=128, type=int)              # batch size
    parser.add_argument('--epsilon_start', default=0.4, type=int)
    parser.add_argument('--reward_coef', default=1, type=float)
    parser.add_argument('--tensorboard', default=False, type=bool)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument("--save_data", default=True, type=bool)             # loss , reward
    parser.add_argument("--save_model", default=True, type=bool)            # model
    parser.add_argument("--load_model", default=False, type=bool)           # load model
    parser.add_argument("--load_model_name", default='20211121_180048_model.pth', type=str)       # load model

    args = parser.parse_args()
    main(args)