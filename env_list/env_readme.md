conda 環境匯出/安裝程式庫匯出到 environment.yaml 和 requirements.txt / pip 和 conda 安裝指定版本的函式庫

1. conda 匯出目前環境庫版本
    conda env export > environment.yaml 或 conda list -e > requirements.txt 

2. conda 導入庫版本進行安裝
    環境會被保存在 environment.yaml 檔案中, 當我們想要再次創建該環境, 或根據別人提供的 .yaml 檔案復現環境時, 指令如下:
    conda env create -f environment.yaml 或 conda install --yes --file requirements.txt

如果原環境中既有 conda 安裝的庫, 也有 pip 安裝的庫, 那麼移植過來的環境只是安裝了你原來環境裡用 conda install 等命令直接安裝的包, 你用 pip 之類裝的東西沒有移植過來, 需要你重新安裝

1. pip 匯出安裝的庫到 requirements.txt
    pip freeze > requirements.txt

2. pip 導入 requirements.txt 中列出的庫安裝到新環境
    pip install -r requirements.txt 或 pip3 install -U pip && pip3 install -r requirements.txt 