import os
import cv2
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm

def main():
    # --- 設定 ---
    # 你的模型產出的圖片資料夾
    PRED_DIR = 'data/competition/upscaled_preds'
    # 輸出的 CSV 檔名
    OUTPUT_CSV = 'submission.csv'
    
    print(f"🚀 Preparing to convert images from {PRED_DIR} to CSV...")

    # 1. 檢查資料夾
    if not os.path.exists(PRED_DIR):
        print(f"❌ 錯誤: 找不到預測結果資料夾: {PRED_DIR}")
        print("💡 請先執行: python -m src.infer_kaggle")
        return

    # 搜尋所有圖片
    extensions = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(glob(os.path.join(PRED_DIR, ext)))
    
    # 重要：依照檔名排序，確保提交順序正確
    files = sorted(files)
    
    if len(files) == 0:
        print("❌ 資料夾是空的，沒有圖片可以轉換。")
        return

    print(f"📂 Found {len(files)} images. Processing...")

    # 2. 開始轉換
    data_list = []
    
    for path in tqdm(files):
        filename = os.path.basename(path)
        
        # 讀取圖片
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Warning: 無法讀取 {filename}")
            continue
            
        # 轉 RGB (確保顏色正確)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # --- 關鍵：將圖片攤平 ---
        # 許多 Kaggle 影像競賽要求將 (H, W, C) 的陣列攤平成一維 (H*W*C)
        # 例如: 256*256*3 = 196608 個數值
        flatten_pixels = img.flatten()
        
        # 這裡示範最常見的格式：
        # 如果比賽要求每個像素一個欄位，這會產生巨大的 CSV (不建議直接用 Excel 開)
        # 如果比賽要求 "Id, Predicted" (字串格式)，請用下面這行：
        # prediction_str = ' '.join(map(str, flatten_pixels))
        
        # 我們先假設是「每個 Row 代表一張圖，包含 ID」
        # 你可能需要根據官方 scripts/official/csv_ify.ipynb 微調這裡
        entry = {
            'Id': filename,                 # 圖片 ID
            'Predicted': flatten_pixels     # 這裡先存 array，後面 pandas 會處理
            # 如果需要字串格式，改用: 'Predicted': ' '.join(map(str, flatten_pixels))
        }
        data_list.append(entry)

    # 3. 轉成 DataFrame 並存檔
    df = pd.DataFrame(data_list)
    
    # 如果像素是分開的欄位 (比較少見，因為檔案會超大)，通常是存成字串或特定格式
    # 這裡我們假設只需要 ID 和 內容
    
    print(f"💾 Saving to {OUTPUT_CSV} ...")
    df.to_csv(OUTPUT_CSV, index=False)
    
    print("🎉 Done! Submission file generated.")
    print("⚠️ 注意：請務必對照官方提供的 'csv_ify.ipynb' 確認欄位名稱 (Header) 是否正確！")

if __name__ == "__main__":
    main()