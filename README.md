# Tomofun 狗音辨識 AI 百萬挑戰賽 - 酒桃郎跟他的貓

實驗詳細步驟請參考: Tomofun狗音辨識AI百萬挑戰賽_初賽賽後報告.docx



# 實驗設備:
．Windows 10 Home Edition

．CPU: i9-10980HK CPU 2.40GHz 3.10 GHz

．GPU: RTX 3080 16GB Laptop

．RAM: 64GB 



# 環境 :

．Python 3.6

．TensorFlow_gpu 2.5.0

．TensorFlow_Addons 0.13.0

．Librosa 0.8.0

．Scikit-learn 0.23.1

．Audionmentations

．Opencv-python 4.5.2.54

．MixupGenerator

# 程式碼:


．train 資料夾:1200筆比賽提供的 train 音檔

．private_test 資料夾 : 20000筆比賽提供的 private_leaderboard 音檔

．public_test 資料夾 : 10000筆比賽提供的 public_leaderboard 音檔

          (以上比賽所提供的訓練資料基於保密協議無法提供)

．mp 資料夾 : MixGenerator套件

．weight 資料夾 : 模型儲存的權重

．preprocess.py : 前處理程式碼，裡面包括建立的訓練模型

．train.py : 訓練模型的所使用的程式碼

．submit.py : 使用訓練好的模型預測 public_test & private_test 音檔，並輸出 final_submission.csv

．efficientnetb0_notop.h5 : EfficientNet B0 NoisyStudent weight

．final_submission.csv :最終上傳 Private_Leaderboard 的預測結果


# 使用 Anaconda 創建環境:

選擇對應的作業系統下載 Andconda:

 https://www.anaconda.com/products/individual

安裝完成後，開啟Andaconda Powershell Prompt，輸入以下指令創建環境:

```
conda create --name tomofun_27th python=3.6 anaconda
```

輸入後會出現是否創建環境，輸入 y 建立:

```
Proceed ([y]/n)?
```
創建完成後輸入以下指令啟動環境:

```
conda activate tomofun_27th
```

將根目錄切換到程式碼相對應的資料夾:
```
cd \資料夾的儲存位置\first_round\
```

輸入以下指令安裝套件:
```
pip install -r requirements.txt
```



# 使用訓練好的模型進行預測 :
以上環境安裝完成後，在Powershell prompt輸入以下指令便可執行模型預測:
```
python submit.py
```

讀取已訓練好的模型來預測 public_test & private_test，輸出 final_submissoin.csv，此 csv 為 Private LearderBoard 的最終上傳成績，執行時間約15分鐘。

# 訓練模型:

在Powershell prompt輸入以下指令:
```
python train.py
```
使用 RTX 3080 16GB Laptop訓練時間約 5 小時，訓練好的模型權重會儲存在 weight 資料夾中。重新訓練會覆蓋掉原本已訓練好的模型權重，建議備份 weight 資料夾後執行。執行完 train.py 之後重新執行 submit.py 便可使用重新訓練的模型產生 final_submission.csv。

