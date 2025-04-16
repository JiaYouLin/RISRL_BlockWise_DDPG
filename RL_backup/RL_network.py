import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# 定義神經網路，繼承自 nn.Module
class DQN(nn.Module):
    def __init__(self, dim_observation, dim_action):
        super(DQN, self).__init__()                         
        th.set_printoptions(profile="full")                 # 完整顯示(debug看數值用)
        
        # 神經網路共四層:128,64,32,32
        self.fc1 = nn.Linear(dim_observation* 10, 128)      # 設定神經網路的各層
        self.fc1.weight.data.normal_(0, 0.1)                # 初始化權重
        self.fc2 = nn.Linear(128, 64)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(64, 32)
        self.fc3.weight.data.normal_(0, 0.1)
        self.fc4 = nn.Linear(32, 32)
        self.fc4.weight.data.normal_(0, 0.1)
        self.out_msg1 = nn.Linear(32, dim_action)
        self.out_msg2 = nn.Linear(32, dim_action)
        self.out_msg3 = nn.Linear(32, dim_action)
        self.out_msg4 = nn.Linear(32, dim_action)
        self.out_msg5 = nn.Linear(32, dim_action)
        
        
        self.out_msg1.weight.data.normal_(0, 0.1)
        self.out_msg2.weight.data.normal_(0, 0.1)
        self.out_msg3.weight.data.normal_(0, 0.1)
        self.out_msg4.weight.data.normal_(0, 0.1)
        self.out_msg5.weight.data.normal_(0, 0.1)


    # 因為我們有5個output(Q-value)因此總共有5個
    # 定義神經網路的前向傳播過程，並返回結果
    def forward(self, x): 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        result_msg1 = self.out_msg1(x)
        result_msg2 = self.out_msg2(x)
        result_msg3 = self.out_msg3(x)
        result_msg4 = self.out_msg4(x)
        result_msg5 = self.out_msg5(x)

        
        return result_msg1,result_msg2,result_msg3,result_msg4,result_msg5 
        

# 將訓練之神經網路儲存，用於保存模型
def save_model(ck_model,ck_optimizer,ck_loss,dt):
    #th.save(model_actor.state_dict(), dt +'_model.pth')
    #model_actor = DQN(6,16)           # dim_observation=6, dim_action=16

    # 保存模型的狀態字典、優化器的狀態字典和損失
    th.save({
            'model_state_dict': ck_model.state_dict(),
            'optimizer_state_dict': ck_optimizer.state_dict(),
            'loss': ck_loss,
            }, dt +'_model.pth')