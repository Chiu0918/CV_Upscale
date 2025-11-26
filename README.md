# 📘 CV Upscale — 4× Super-Resolution 專案（開發中）

本專案為Kaggle競賽2024-Upscale，目標是將 **64×64** 低解析度影像重建成 **256×256** 高解析度影像（4× 放大）。
目前已完成官方降採樣流程與資料前處理，後續會依序實作 SRCNN、U-Net、評估指標與 Kaggle 提交流程。

---

# 📂 專案結構（現階段）

```
cv_2024_upscale/
├─ data/                        # 資料目錄（未納入 Git）
│   ├─ train_hr/                # 高解析度訓練影像（256×256）
│   ├─ train_lr/                # 使用官方方式降採樣後的低解析度影像（64×64）
│   └─ competition/             # Kaggle 官方資料
│       ├─ downscaled/          # 官方給的 64×64 測試影像
│       └─ csv/                 # 官方 baseline 的 CSV
│
├─ scripts/
│   ├─ official/                # Kaggle 官方提供的腳本
│   │   ├─ downscale_all.py
│   │   ├─ upscale_all.py
│   │   ├─ csv_ify.ipynb
│   │   └─ down-scale.ipynb
│   └─ tools/
│       └─ prepare_train_data.py  # 批次產生訓練用 LR 影像
│
├─ src/
│   ├─ data/
│   │   ├─ degrade.py           # 官方降採樣的 Python 封裝
│   │   └─ dataset_pairs.py     # PyTorch Dataset（LR/HR 成對載入）
│   ├─ models/                  # 模型（SRCNN / U-Net）
│   │   ├─ srcnn.py
│   │   └─ unet_sr.py
│   ├─ train.py                 # 訓練主程式（待擴充）
│   ├─ eval.py                  # 評估程式（PSNR / SSIM）
│   ├─ infer_kaggle.py          # Kaggle 推論
│   └─ to_csv.py                # 轉 CSV（提交格式）
│
├─ notebooks/                   # 測試＆分析 Notebook
│
├─ models_ckpt/                 # 模型權重（未納入 Git）
│
├─ README.md
└─ environment.yml
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

---

# 📥 安裝與環境設定

## 1️⃣ 取得專案（Git Clone）

請先安裝 Git，然後在任意資料夾執行：

```bash
git clone https://github.com/aceyang108/CV_Upscale.git
cd CV_Upscale
```

若你是團隊成員，建議 fork 後以 Pull Request 方式提交變更。

---

## 2️⃣ 建立 conda 環境（建議使用）

本專案使用 `environment.yml` 管理依賴套件。

### 建立環境：

```bash
conda env create -f environment.yml
```

### 啟動環境：

```bash
conda activate upsr
```

---

## 3️⃣ 若希望手動安裝（快速測試資料前處理）

如果你只想執行資料產生（官方降採樣）工具而不需要完整機器學習套件：

```bash
pip install opencv-python numpy
```

---

## 4️⃣ 專案資料夾注意事項

本專案採用 `.gitignore` 排除大型檔案與資料集，因此：

* `data/` 資料夾 **不會被 Git 同步**
* 請自行下載或產生 `train_hr/`、`train_lr/` 與 `competition/downscaled/`
* `models_ckpt/` 也不會進入版本控制

---

## 5️⃣ 產生訓練用低解析度（LR）影像

將所有 256×256 HR 圖片放入：

```
data/train_hr/
```

執行：(修正成可接受jpg、png、jpeg，並加上data Augmentation)

```bash
python -m scripts.tools.prepare_train_data
```

生成的 LR 影像（64×64）會存至：

```
data/train_lr/
```

---

## 6. 衡量用腳本

check_data_visual.py: 檢查確認產生的lr圖片。
check_model_result.py: 生成lr|model_result|hr圖片，用視覺展示模型結果，可自由選擇哪個model。
```bash
python check_model_result.py
```
src/eval.py: 用PSNR和SSIM數值展示單個模型結果，可自由選擇哪個model。
```bash
python -m src.eval
```
src/compare_to_baseline.py: 比較Bicubic / Nearest 和這次實作的SRCNN、U-Net，用PSNR和SSIM比較。
```bash
python -m src.compare_to_baseline
```
src/to_csv.py:生成繳交上kaggle的csv
---

## 7.Model
實作src/models/scrnn.py，並用src/train_for_srcnn.py訓練，並把結果存在models_ckpt。
```bash
python -m src.train_for_srcnn
```
實作src/models/unet_sr.py，並用src/train_for_unet.py訓練，並把結果存在models_ckpt。
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

* [x] 完整 `train.py`：epoch、log、存最佳模型

### 🔹 評估

* [x] PSNR / SSIM 計算
* [x] Bicubic / Nearest / SRCNN / U-Net 比較

### Model upgrade
* [ ] 解決 U-Net 圖片 Over-smoothing 問題。
* [ ] 引入今天學到課程內的 Residual 與 Attention 來疊更深。
* [ ] 做成 Patch Training。

### 🔹 Kaggle

* [ ] 對 `data/competition/downscaled/` 做推論
* [ ] 產生 `upscaled_model.csv` 並提交

---

# 📌 作者 / 貢獻者

* Chiu0918
* aceyang108
* 2024–2025

