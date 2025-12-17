# 📘 CV Upscale — 4× Super-Resolution 專案（開發中）

本專案為Kaggle競賽2024-Upscale，目標是將 **64×64** 低解析度影像重建成 **256×256** 高解析度影像（4× 放大）。
目前已完成官方降採樣流程、資料前處理、SRCNN、U-Net、評估指標、patch training功能，後續會依序實作模型優化、Kaggle 提交流程。
---


# 📂 專案結構（現階段）

> 註：`data/`、`models_ckpt/` 皆未納入 Git，需自行準備。

```text
cv_2024_upscale/
│  .gitignore
│  check_data_visual.py          # 檢查 LR / HR 是否正確對齊的可視化工具
│  check_model_result.py         # 產生 LR | model_result | HR 的對照圖
│  environment.yml
│  README.md
│
├─data/
│  ├─competition/
│  │  ├─csv/                     # 官方 / baseline CSV
│  │  ├─downscaled/              # 官方提供的 64×64 測試影像
│  │  ├─originals_bis/
│  │  ├─upscaled_bicubic/
│  │  └─upscaled_nearest/ 
│  ├─train_hr/                   # 統一採用DIV2K + Flickr2K
│  ├─train_lr/                   # 由 HR 降採樣產生的 LR (64×64)
│  ├─val_hr/                     # 驗證用 HR（由 split_train_val.py 自動產生）
│  └─val_lr/                     # 驗證用 LR（由 split_train_val.py 自動產生）
│
├─logs/
│  └─ <exp_name>/               # 訓練日誌（train_log.csv）
│       └─ train_log.csv        # 每 epoch 的 train / val loss 與 learning rate
│
├─models_ckpt/
│  └─ <exp_name>_best.pth     # 驗證集 Loss 最佳
│  └─ <exp_name>_epochXX.pth  # 週期性存檔
│  └─ <exp_name>_final.pth    # 最終模型
│
├─notebooks/
│  ├─0_data_check.ipynb          # 確認資料與對應關係
│  ├─1_baseline_analysis.ipynb   # Bicubic / Nearest 等 baseline 分析
│  ├─2_model_evaluation.ipynb     # 模型評估（PSNR / SSIM、可視化）
│  └─3_training_experiments.ipynb# 實驗記錄與不同訓練設定比較
│
├─scripts/
│  ├─official/
│  │  ├─csv_ify.ipynb
│  │  ├─down-scale.ipynb
│  │  ├─downscale_all.py
│  │  └─upscale_all.py
│  └─tools/
│     └─prepare_train_data.py    # 批次產生訓練用 LR 影像（支援 jpg/png/jpeg）
│     └─split_train_val.py       # 將資料分割為 train / val
│
└─src/
   │  compare_to_baseline.py     # 比較 Bicubic / Nearest / SRCNN / U-Net / SRGAN / EDSR
   │  infer_kaggle.py            # 對 Kaggle 測試集做推論
   │  to_csv.py                  # 產生提交用 CSV
   │  train.py                   # （舊版）訓練入口，已被專用 train_for_* 取代
   │  train_for_srcnn.py         # SRCNN 訓練腳本（支援 TrainConfig）
   │  train_for_unet.py          # U-Net 訓練腳本（支援 TrainConfig + Patch）
   │  train_for_edsr.py          # EDSR 訓練腳本（不支援 TrainConfig + Patch）若只修改一參數，沒必要
   │  train_for_srgan.py         # SRGAN 訓練腳本（不支援 TrainConfig + Patch）若只修改一參數，沒必要
   │  utils.py                   # 共同工具函式
   │
   ├─config/
   │  └─train_config.py          # TrainConfig：集中管理訓練超參數與命名規則
   │
   ├─data/
   │  ├─degrade.py               # 官方降採樣邏輯 Python 封裝
   │  └─dataset_pairs.py         # PyTorch Dataset（支援成對 / augmentation / patch）
   │
   └─models/
      ├─srcnn.py                 # SRCNN 模型
      ├─edsr.py                  # EDSR 模型 (No-BN ResNet)
      ├─srgan.py                 # SRGAN 模型 (Generator + Discriminator)
      ├─unet_v1.py               # U-Net 模型 (記錄用)
      ├─unet_v2.py               # U-Net 模型 (記錄用，Res-Attn版)
      └─unet_sr.py               # U-Net SR 模型（目前是unet_v2，）
```

---

# 🧪 已完成功能（目前進度）

### ✔ 官方降採樣邏輯封裝（`src/data/degrade.py`）

使用 Kaggle 官方提供的降採樣方式：

```
img[::4, ::4, :]
```

確保訓練資料與競賽測試資料的分布一致。

---

### ✔ 批次產生訓練用 LR 影像（`scripts/tools/prepare_train_data.py`）

用途：將 `data/train_hr/` 裡的 256×256 HR 影像依官方方法轉成 64×64 LR。

**使用方式：**

1. 將高解析度 PNG 放入：

```
data/train_hr/
```

2. 執行指令：

```bash
python -m scripts.tools.prepare_train_data
```

3. 輸出會自動存到：

```
data/train_lr/
```

---

✔ PyTorch Dataset（src/data/dataset_pairs.py）

可載入 (LR, HR) 成對影像

支援 transform

回傳 tensor（C×H×W）

支援 Patch Training（patch_size）

自動根據設定決定輸出 full image 或 patch

確保 LR/HR 尺寸符合 scale factor

---

### ✔ Train/Val Split（`split_train_val.py`）

```bash
python -m scripts.tools.split_train_val

```

產生：

```
data/train_hr, train_lr
data/val_hr, val_lr

```

---

# 📥 安裝與環境設定

## 1️⃣ 取得專案（Git Clone）

請先安裝 Git，然後在任意資料夾執行：

```bash
git clone https://github.com/Chiu0918/CV_Upscale
cd CV_Upscale
```

若你是團隊成員，建議 fork 後以 Pull Request 的方式提交變更。

---

## 2️⃣ 建立 Conda 環境（建議）

本專案使用 `environment.yml` 管理所有依賴套件。

**建立環境：**

```bash
conda env create -f environment.yml
```

**啟動環境：**

```bash
conda activate upsr
```

---

## 3️⃣ 若只需執行資料前處理（最小安裝）

若你暫時 **不需要訓練模型**，只想快速產生 LR 影像，可僅安裝：

```bash
pip install opencv-python numpy
```

---

## 4️⃣ 專案資料夾注意事項（重要）

由於 `.gitignore` 已排除大量資料，因此以下資料夾 **不會被 Git 同步**：

* `data/` — 需自行準備

  * `train_hr/`（256×256 高解析度）
  * `train_lr/`（由 HR 降採樣而來）
  * `competition/downscaled/`（Kaggle 官方 64×64 測試影像）
* `models_ckpt/` — 存放訓練好的模型權重

若你是首次抓專案，請自行建立上述資料夾或放入相對應的資料。

---

## 5️⃣ 準備資料集

本專案採用DIV2K( https://www.kaggle.com/datasets/soumikrakshit/div2k-high-resolution-images )+Flickr2K( https://www.kaggle.com/datasets/daehoyang/flickr2k?select=Flickr2K )作為主要訓練資料。請將下載後的 HR (高解析度) 圖片放入：

```text
data/train_hr/
```

然後執行官方降採樣封裝（支援 `jpg / png / jpeg`，並內建 Data Augmentation）：

```bash
python -m scripts.tools.prepare_train_data
```

程式會自動產生對應的 64×64 LR 影像到：

```text
data/train_lr/
```

---

## 6️⃣ 評估與可視化腳本

### 🔹 檢查資料是否正確

`check_data_visual.py`
顯示 HR 與 LR 的對照圖，用於確認資料配對無誤。

---

### 🔹 檢視模型輸出結果

`check_model_result.py`
現在支援透過參數指定模型架構與權重，無需修改程式碼。
參數說明：
-m:  (unet, edsr, srgan, srcnn, esrgan)
-c: (.pth)

```bash
# 測試 U-Net
python check_model_result.py -m unet -c models_ckpt/unet_best.pth

# 測試 EDSR
python check_model_result.py -m edsr -c models_ckpt/edsr_best.pth

# 測試 SRGAN
python check_model_result.py -m srgan -c models_ckpt/srgan_G.pth
```

---

### 🔹 與傳統插值法比較（Bicubic / Nearest）

`src/compare_to_baseline.py`
比較不同模型與 Bicubic / Nearest 的 PSNR 與 SSIM 指標。支援一次輸入多個模型進行比較，有用random。

```bash
python -m src.compare_to_baseline \
    --unet models_ckpt/unet_v1_best.pth \
    --edsr models_ckpt/edsr_baseline_best.pth \
    --srgan models_ckpt/srgan_best.pth
```

---

### 🔹 產生 Kaggle 提交檔

`src/to_csv.py`
將模型輸出轉成 Kaggle 需要的 CSV 格式：

```bash
python -m src.to_csv
```

---

# 7️⃣ 模型訓練

## 🔹 SRCNN

```bash
python -m src.train_for_srcnn

```

### 🔹 訓練 U-Net SR

```bash
python -m src.train_for_unet
```

### 🔹 訓練 EDSR

```bash
python -m src.train_for_edsr
```

### 🔹 訓練 SRGAN 

```bash
python -m src.train_for_srgan
```

所有訓練腳本支援：

- Patch Training
- Train/Val 分割
- Best Model 自動儲存：`_best.pth`
- 週期性存檔：`_epochX.pth`
- 最終模型：`_final.pth`
- 訓練日誌：`logs/<exp_name>/train_log.csv`

### 📈 訓練紀錄與最佳模型

目前訓練腳本（`train_for_srcnn.py`, `train_for_unet.py`）皆支援：

- 自動儲存最佳模型：`<exp_name>_best.pth`
- 週期性模型：`_epochX.pth`
- 最終模型：`_final.pth`
- 訓練過程自動寫入 `logs/<exp_name>/train_log.csv`

train_log.csv 內容包含：

| epoch | train_loss | val_loss | learning_rate |
|-------|------------|----------|----------------|
| 1 | ... | ... | ... |
| 2 | ... | ... | ... |

你可以在 notebook 讀取並繪製 loss 曲線：

```python
import pandas as pd
df = pd.read_csv("logs/<exp_name>/train_log.csv")
df[["train_loss", "val_loss"]].plot()
```

---

# 已完成

### 🔹 資料處理

* [x] 實作 `dataset_pairs.py`：讀取 `(LR, HR)` 成為 PyTorch Dataset
* [x] check_data_visual.py: 確認hr vs lr

### 🔹 模型

* [x] SRCNN baseline（對照 Bicubic / Nearest）
* [x] U-Net SR 模型

### 🔹 訓練

* [x] 完整 `train_fot_srcnn.py`, `train_for_unet.py`：epoch、log、存最佳模型

### 🔹 評估

* [x] PSNR / SSIM 計算
* [x] Bicubic / Nearest / SRCNN / U-Net 比較

### Model upgrade
* [x] 引入今天學到課程內的 Residual 與 Attention 來疊更深。
* [x] 做成 Patch Training。

# 未完成 

* [ ] SRGAN 優化：調整 Loss 權重以平衡偽影問題。
* [ ] Kaggle 提交：針對 data/competition/downscaled/ 進行全圖推論並產生 CSV。
* [ ] Ablation Study：比較不同 Patch Size 的影響。

---

# 📌 作者 / 貢獻者

* Chiu0918
* aceyang108
* 2024–2025

