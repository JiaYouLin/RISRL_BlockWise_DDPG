# RISRL_BlockWise_DDPG

Retrieve the code from this version of the RIS_RL repository (a8ece9a376e954ff917ab7b645b32aaf7c948493), link:
https://github.com/JiaYouLin/RIS_RL/tree/a8ece9a376e954ff917ab7b645b32aaf7c948493. 

This section of the code represents the official release.

# Description 

1. Action 為每個 step 調整全部 Block 的角度，而不是一次調整所有 element 或只調整單一個 element。注意，是以 Block 危機底座調整。舉例: 以總體 `8*8=64` 個 element來說，以 Block `G = 1*8` 個 element，也就是一列。在一個 Block 中，每個 element 的 phase 是一致，一旦 Action 決定了該 Block 的 Phase，則 Block 中每個 element phase 都會是一樣的 Phase。
2. 在 `choose_action` 中，這裡是選擇所有 Block 的 Phase，因為每個 Block 有 1-bits (2 個 Phase) 可以選擇，所以在每個 Block 的 Q-value 也會有四種可以選擇，可選最大的 Q-value 作為本次 Action；或是 random 選擇一個組合作為 Action。
3. 將 Testing 直接接在 Training 結束後，不額外拉出來寫。

# PPO Design (尚未修正)
1. Action Space (動作空間): 每個 element 獨立選擇 phase shift。
   
    將 action space 設計為 64 維，每個 element 選擇 4 個 phase shift 之一。

    這樣 Action Space 為 `64 x 4`，即 64 個 RIS 元件，每個有 4 個選項，而不是一次選擇整片 RIS。

2. State Space (狀態空間): RIS 的 64 個 element 當前 phase shift。

    UE SINR (假設 1 個 UE)，`64 + 1 = 65` 維。

3. Reward (獎勵函數): 使用 UE Datarate (可用 Shannon Capacity)。

    ```
    reward = sum(log2(1 + SINR))
    ```

# PPO Architecture (尚未修正)
1. Actor-Critic 架構

    - Actor ($\pi$): 決定每個 RIS element 的 phase shift (64 維離散選擇)。
    - Critic ($V$): 評估當前狀態的價值 (State-Value Function)。
    - PPO 會透過 Clipped Objective 限制更新幅度，穩定訓練。
2. Action 空間(離散版)

    - action_space = $64 \times 4$
    - Output logits shape = `(batch_size, 64, 4)`
    - 用 `Categorical` 分布來選擇每個 RIS element 的 phase shift。

# Note

1. 已改好 element phase 生成為 1-bit 離散值，所以只要在 main.py 中的 parameter args seeting 中設置好，即會自動切換與計算。
2. 已釐清不須理會複數。

   因為一開始宣逸只是生成 0~2pi 的 random (continues)值。

   之後使用會是複數(實部+虛部)是因為他使用 tensor 儲存(tensor 會自動轉換為複數(實部+虛部)型態儲存)。

   隨後使用 tensor (複數型態)計算 SINR。

   所以，簡單來說只要我調整的時候使用原始生成的小數點值，接著計算 SINR 時再以 tensor 型態儲存。

   給入計算 SINR 的 function，這樣就沒問題了。
3. 至於調整 element phase 的 function，這在程式碼中也是直接對原始生成的小數點值進行操作，所以會是以小數點形式調整。

   與第 2 點同理，隨後調整完畢計算 SINR 前，將調整後整體的 element phase 以 tensor 儲存，給入計算 SINR 的 function，這樣就沒問題了。

# Code List

1. [main.py](./main.py): 主程式
2. [RISdata.py](./RISdata.py): 初始化 element phase
3. [env.py](./env.py): RL 環境定義，包括 next state、next reward
4. [agent.py](./agent.py): agent 做的事情以及 model 資料，包括選擇 action、memory、train、獲取 batches
5. [network,py](./network.py): RL network 架構、儲存 model
6. [plot_loss_reward.py](./plot_loss_and_reward.py): 畫 training loss、reward 折線圖
7. [save.py](./save.py): 儲存 action、Q-Value 值
8. [output.py](./output.py): 儲存 training 和 testing 過程中的資料到 CSV，包括 Episode(timeslot)、agent、step、temp_step、reward、loss、action、Q_value
9. [utils.py](./utils.py): RIS 模擬場景配置
10. [environment.py](./environment.py): RIS 模擬場景的環境模擬
11. [channel.py](./channel.py): RIS 模擬場景的channel模擬

# 執行指令

1. Training model

    ```
    python main.py
    ```

2. Load model

    ```
    python main.py --load_model True --load_model_name ./model/20241002_113848_model.pth
    ```

# GitHub 上傳忽略不必要的檔案
在 .gitignore 文件中指定要忽略的檔案和資料夾，Git 在追蹤時會自動忽略這些項目，這樣它們就不會出現在未提交的變更中，也不會被推送到遠端儲存庫。

1. 建立或編輯 .gitignore
    
    在你的專案根目錄中應該有一個 .gitignore 檔案。如果沒有，請手動創建一個。

        ```
        vim .gitignore
        ```

2. 添加要忽略的檔案或資料夾
    
    在 .gitignore 文件中添加你想要忽略的檔案或資料夾路徑。

        ```
        忽略 Python 的編譯檔案
        __pycache__/
        *.pyc

        忽略日誌檔案
        *.log

        忽略臨時檔案和目錄
        temp/
        *.tmp

        忽略特定檔案
        config.txt

        # 忽略操作系統生成的文件
        .DS_Store
        Thumbs.db
        ```

3. 保存 .gitignore 文件

        ```
        PRESS ":", then input "wq", ENTER
        ```

4. 確認忽略生效

    添加到 .gitignore 的項目將從此不再被 Git 追蹤。如果這些檔案或資料夾已經被 Git 追蹤，則需要從版本控制中移除它們（但不會刪除本地檔案）。

    * 移除已追蹤的檔案（但保留本地檔案）: 
        
        --cached 的意思是只從 Git 的版本控制中移除檔案，而不是刪除本地檔案，-r 表示遞迴操作，會移除整個資料夾的所有內容。

            ```
            git rm --cached <file_or_directory>
            ```
        
        例如，忽略 __pycache__ 資料夾

            ```
            git rm --cached -r __pycache__
            ```

5. 確認 .gitignore 生效

    執行完 git rm --cached 之後，檔案將不再受 Git 版本控制。你應該會看到 __pycache__ 和其他在 .gitignore 中指定的檔案已經顯示為未追蹤（untracked），不再顯示在 Git 的變更列表中。

        ```
        git status
        ```

    或是可以查看有哪些檔案被 Git 追蹤

        ```
        git ls-files
        ```

6. 提交 .gitignore 文件
    
    你應該會看到 __pycache__ 和其他在 .gitignore 中指定的檔案已經顯示為未追蹤（untracked），不再顯示在 Git 的變更列表中。將 .gitignore 文件提交到版本控制中，這樣它會隨專案一同被推送到 GitHub。

        ```
        git add .gitignore
        git commit -m "add gitignore"
        git push origin main
        ```

# GitHub 上傳:
1. 新增檔案

    ```
    git add .
    ```

2. 新增註解
    ```
    git commit --allow-empty-message -m ""
    ```
    
    or

    ```
    git commit -m "message"
    ```
3. 推送
    ```
    git push origin main
    ```

    or

    ```
    git push
    ```
