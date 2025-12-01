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
│  ├─train_hr/                   # 訓練用 HR (256×256)
│  ├─train_lr/                   # 由 HR 降採樣產生的 LR (64×64)
│  ├─val_hr/                     # （預留）驗證用 HR
│  └─val_lr/                     # （預留）驗證用 LR
│
├─models_ckpt/                   # 訓練好的權重（未納入 Git）
│  ├─srcnn_*.pth                 # 各種 SRCNN 實驗（含 patch / full）
│  └─unet_*.pth                  # 各種 U-Net 實驗（含 patch / full）
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
│
└─src/
   │  compare_to_baseline.py     # 比較 Bicubic / Nearest / SRCNN / U-Net
   │  eval.py                    # 評估腳本（PSNR / SSIM，支援指定 checkpoint）
   │  infer_kaggle.py            # 對 Kaggle 測試集做推論
   │  to_csv.py                  # 產生提交用 CSV
   │  train.py                   # （舊版）訓練入口，已被專用 train_for_* 取代
   │  train_for_srcnn.py         # SRCNN 訓練腳本（支援 TrainConfig）
   │  train_for_unet.py          # U-Net 訓練腳本（支援 TrainConfig + Patch）
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
      └─unet_sr.py               # U-Net SR 模型（含 encoder/decoder 結構）
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

# 📥 安裝與環境設定

## 1️⃣ 取得專案（Git Clone）

請先安裝 Git，然後在任意資料夾執行：

```bash
git clone https://github.com/aceyang108/CV_Upscale.git
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

## 5️⃣ 產生訓練用低解析度（LR）影像

將所有 **256×256 HR** 圖檔放入：

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
產生 **LR | model_result | HR** 的三合一對照圖，可手動選擇要測試的模型：

```bash
python check_model_result.py
```

---

### 🔹 單模型 PSNR / SSIM 評估

`src/eval.py`
輸出某個模型在整個資料集的 PSNR / SSIM：

```bash
python -m src.eval
```

---

### 🔹 與傳統插值法比較（Bicubic / Nearest）

`src/compare_to_baseline.py`
比較 Bicubic / Nearest 與 SRCNN、U-Net 的 PSNR / SSIM 表現：

```bash
python -m src.compare_to_baseline
```

---

### 🔹 產生 Kaggle 提交檔

`src/to_csv.py`
將模型輸出轉成 Kaggle 需要的 CSV 格式：

```bash
python -m src.to_csv
```

---

## 7️⃣ 模型訓練（SRCNN / U-Net）

### 🔹 訓練 SRCNN

模型結構：`src/models/srcnn.py`
訓練腳本：`src/train_for_srcnn.py`

```bash
python -m src.train_for_srcnn
```

訓練後模型會存入：

```text
models_ckpt/
```

---

### 🔹 訓練 U-Net SR

模型結構：`src/models/unet_sr.py`
訓練腳本：`src/train_for_unet.py`

```bash
python -m src.train_for_unet
```
---

# 🔜 TODO（接下來的開發計畫）

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
* [ ] 解決 U-Net 圖片 Over-smoothing 問題。
* [ ] 引入今天學到課程內的 Residual 與 Attention 來疊更深。
* [x] 做成 Patch Training。

### 🔹 Kaggle

* [ ] 對 `data/competition/downscaled/` 做推論
* [ ] 產生 `upscaled_model.csv` 並提交

---

# 📌 作者 / 貢獻者

* Chiu0918
* aceyang108
* 2024–2025

