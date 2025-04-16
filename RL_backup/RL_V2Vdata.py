import numpy as np
import matlab.engine
from matlab import double
eng = matlab.engine.start_matlab()                                          # 啟動 Matlab 引擎

def V2Vstate(timeslot,MRB):                                                 # V2V前初始狀態
    
    file_name = "solution/timeSlot_%d/MRB_%d.csv" % (timeslot, MRB)         # 根據 timeslot 和 MRB 設定檔案名
    np.set_printoptions(suppress=True)      # 浮點數輸出的精度位數
    np.set_printoptions(threshold=np.inf)   # 完整顯示

    ##############  計算SINR部分  ############## 
    # 讀取位置、車隊和領導者數據
    getdata_position = np.genfromtxt(file_name, delimiter=',',  usecols=(1,2),  
                            encoding='UTF-8')
    getdata_platoon = np.genfromtxt(file_name, delimiter=',',  usecols=(3),  
                            encoding='UTF-8')
    getdata_is_leader = np.genfromtxt(file_name, delimiter=',',  usecols=(4),  
                            encoding='UTF-8')

    getdata_platoon=np.ravel(getdata_platoon)                               # 轉一維
    getdata_is_leader[np.isnan(getdata_is_leader)] = 0                      # 將nan換成0，處理 getdata_is_leader 中的 nan 值

    leader_total = int(np.add.reduce(getdata_is_leader))                    # 計算leader 總數    
    getdata_leader_position = np.zeros((leader_total,2))                    # 初始化領導者位置

    j=0
    for i in range(0, len(getdata_position)): 
        if getdata_is_leader[i] == 1:                                       # 取得leader position
            #print(getdata_position[i])
            getdata_leader_position[j]=getdata_position[i]
            j += 1

    LeaderTrans = np.zeros(j)
    for z in range(0, j):
        LeaderTrans[z] = 0  

    # 將領導者位置和其他數據轉換為列表
    getdata_position = getdata_position.tolist()
    getdata_platoon = getdata_platoon.tolist()
    getdata_leader_position = getdata_leader_position.tolist()
    LeaderTrans = LeaderTrans.tolist()
    getdata_is_leader = getdata_is_leader.tolist()

    for i in range(0, len(getdata_position)): 
        # print(len(getdata_position))
        # print("i = %d" %i)
        if i<len(getdata_position):
            if getdata_is_leader[i] == 1:                                   #  取得leader position
                del getdata_is_leader[i]
                del getdata_position[i]                                     # 將一般Veh list中刪除leader位置
                del getdata_platoon[i]                                      # 以及Veh platoon list 
                i -= 1

    #print(getdata_position) #VehPos
    #print(getdata_platoon)  #VehLeader
    #print(getdata_is_leader)
    #print(getdata_leader_position) #VehLeaderPos

    # 計算 SINR
    SINR = np.array(eng.run(double(getdata_position),double(getdata_platoon),double(getdata_leader_position),double(LeaderTrans),matlab.double([1])))
    #print(SINR)

    ##############  計算SINR部分  ##############
    

    ##############  處理SINR部分  ##############
    getdata_info = np.genfromtxt(file_name, delimiter=',',  usecols=(1,2,3,4,5,6,7,8,9), encoding='UTF-8')
    getdata_info[np.isnan(getdata_info)] = 0                                # 將nan換成0，讀取環境信息並處理 nan 值

    SINR_data = np.zeros((len(getdata_info),1))                             # 建立SINR data
    platoon_start_id = int(min(getdata_info[:,2]))                          # 確認platoon開始ID(為了判斷array位置)
    for v in range(0, len(getdata_info)): 
        for c in range(0, len(getdata_position)): 
            if  not int(getdata_info[v,3]) and (getdata_position[c][0]==getdata_info[v,0]) and (getdata_position[c][1]==getdata_info[v,1]):
                SINR_data[v]=SINR[int(getdata_info[v,2])-platoon_start_id][c]
            elif int(getdata_info[v,3]):
                SINR_data[v] = np.nan
    #print(SINR_data)
    ##############  處理SINR部分  ##############

    V2V_state = np.concatenate(([getdata_info,SINR_data]),axis=1)           # 將原本環境中的資料與SINR合併
    #V2V_state=np.ravel(V2V_state)           #多維轉一維
    #print(V2V_state)

    return V2V_state

def Transmit_state(timeslot,MRB,LeaderTrans):

    file_name = "solution/timeSlot_%d/MRB_%d.csv" % (timeslot, MRB)         # 根據 timeslot 和 MRB 設定檔案名
    np.set_printoptions(suppress=True)                                      # 浮點數輸出的精度位數
    np.set_printoptions(threshold=np.inf)                                   # 完整顯示

    getdata = np.genfromtxt(file_name, delimiter=',',  usecols=(1,2,3,4,10,11,12,13,14),  
                            encoding='UTF-8')
    getdata[np.isnan(getdata)] = 0                                          # 將nan換成0，讀取數據並處理 nan 值
    prob_data = np.array(getdata)
    #print(getdata)

    '''
    # 只單獨Leader的機率
    a = 0
    for i in range(0, len(getdata)): 
        veh_is_leader = getdata[i-a]  
        if int(veh_is_leader[3]) != 1:                    #  取得Veh是否為leader(不是則刪除)
            getdata = np.delete(getdata, i-a, axis = 0)
            a += 1
    '''

    ##############  計算SINR部分  ##############
    getdata_position = np.genfromtxt(file_name, delimiter=',',  usecols=(1,2),  
                            encoding='UTF-8')
    getdata_platoon = np.genfromtxt(file_name, delimiter=',',  usecols=(3),  
                            encoding='UTF-8')
    getdata_is_leader = np.genfromtxt(file_name, delimiter=',',  usecols=(4),  
                            encoding='UTF-8')

    getdata_platoon=np.ravel(getdata_platoon)                               # 轉一維
    getdata_is_leader[np.isnan(getdata_is_leader)] = 0                      # 將nan換成0，讀取環境信息並處理 nan 值

    leader_total = int(np.add.reduce(getdata_is_leader))                    # leader 總數    
    getdata_leader_position = np.zeros((leader_total,2))

    j=0
    for i in range(0, len(getdata_position)): 
        if getdata_is_leader[i] == 1:                                       # 取得leader position
            #print(getdata_position[i])
            getdata_leader_position[j]=getdata_position[i]
            j += 1
    '''
    LeaderTrans = np.zeros(j)
    for z in range(0, j):
        LeaderTrans[z] = 0  
    '''
    getdata_position = getdata_position.tolist()
    getdata_platoon = getdata_platoon.tolist()
    getdata_leader_position = getdata_leader_position.tolist()
    #LeaderTrans = LeaderTrans.tolist()
    getdata_is_leader = getdata_is_leader.tolist()

    for i in range(0, len(getdata_position)): 
        if i<len(getdata_position):
            if getdata_is_leader[i] == 1:                                   # 取得leader position
                del getdata_is_leader[i]
                del getdata_position[i]                                     # 將一般Veh list中刪除leader位置
                del getdata_platoon[i]                                      # 以及Veh platoon list 
                i -= 1

    SINR = np.array(eng.run(double(getdata_position),double(getdata_platoon),double(getdata_leader_position),double(LeaderTrans),matlab.double([1])))

    ##############  計算SINR部分  ##############
    

    ##############  處理SINR部分  ##############
    getdata_info = np.genfromtxt(file_name, delimiter=',',  usecols=(1,2,3,4,5,6,7,8,9), encoding='UTF-8')
    getdata_info[np.isnan(getdata_info)] = 0                                # 將nan換成0

    SINR_data = np.zeros((len(getdata_info),1))                             # 建立SINR data
    platoon_start_id = int(min(getdata_info[:,2]))                          # 確認platoon開始ID(為了判斷array位置)
    for v in range(0, len(getdata_info)): 
        for c in range(0, len(getdata_position)): 
            if  not int(getdata_info[v,3]) and (getdata_position[c][0]==getdata_info[v,0]) and (getdata_position[c][1]==getdata_info[v,1]):
                SINR_data[v]=SINR[int(getdata_info[v,2])-platoon_start_id][c]
            elif int(getdata_info[v,3]):
                SINR_data[v] = np.nan
    #print(SINR_data)
    ##############  處理SINR部分  ##############


    Trans_state = np.concatenate(([prob_data,SINR_data]),axis=1)            # 將原本環境中的資料與SINR合併
    #print(Trans_state)
    return Trans_state

# 獲取代理總數
def get_agent(timeslot,MRB):
    file_name = "solution/timeSlot_%d/MRB_%d.csv" % (timeslot, MRB)         # 根據 timeslot 和 MRB 設定檔案名
    getdata_is_leader = np.genfromtxt(file_name, delimiter=',',  usecols=(4),  
                            encoding='UTF-8')

    getdata_is_leader[np.isnan(getdata_is_leader)] = 0                      # 將nan換成0，讀取 is_leader 數據並處理 nan 值

    agent_total = int(np.add.reduce(getdata_is_leader))                     # 計算leader總數    
    print(agent_total)
    return agent_total                                                      # 返回領導者總數


# 測試 V2Vstate 方法，印出結果
if __name__ == '__main__':
    a = V2Vstate(1,20)          #for test
    #a = Leader_Prob(1,20)          #for test
    #a = get_agent(1,20)          #for test
    print(a)


'''''''''''''''''''''''''''''''''''''''''''''''''''
V2Vstate data:
    position_x, position_y, platoon_ID, is_leader, msg1, msg2, msg3, msg4, msg5, SINR

'''''''''''''''''''''''''''''''''''''''''''''''''''