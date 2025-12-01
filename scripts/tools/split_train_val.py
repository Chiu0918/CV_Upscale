import os
import random
import argparse
import shutil


def collect_pairs(hr_dir: str, lr_dir: str):
    """
    蒐集 (HR, LR) 成對檔案路徑。
    目前只掃描 hr_dir「根目錄」的檔案，不會遞迴子資料夾。
    """
    if not os.path.isdir(hr_dir):
        raise FileNotFoundError(f"找不到 HR 資料夾: {hr_dir}")
    if not os.path.isdir(lr_dir):
        raise FileNotFoundError(f"找不到 LR 資料夾: {lr_dir}")

    hr_files = [
        f for f in os.listdir(hr_dir)
        if os.path.isfile(os.path.join(hr_dir, f))
    ]

    pairs = []
    for name in hr_files:
        hr_path = os.path.join(hr_dir, name)
        lr_path = os.path.join(lr_dir, name)
        if os.path.exists(lr_path):
            pairs.append((hr_path, lr_path))
        else:
            print(f"[警告] 找不到對應的 LR 檔案，略過: {name}")
    return pairs


def split_train_val(
    hr_dir: str,
    lr_dir: str,
    val_hr_dir: str,
    val_lr_dir: str,
    val_ratio: float = 0.1,
    seed: int = 42,
):
    """
    將一部分 (HR, LR) 成對影像，從 train_* 移動到 val_*。
    """

    os.makedirs(val_hr_dir, exist_ok=True)
    os.makedirs(val_lr_dir, exist_ok=True)

    # 為了避免重複切分，如果 val_* 已經不是空的，就直接跳出
    if os.listdir(val_hr_dir) or os.listdir(val_lr_dir):
        print("❌ 偵測到 val_hr 或 val_lr 不是空的，為避免重複切分，腳本已停止。")
        print("   若要重新切分，請先手動清空 data/val_hr 與 data/val_lr。")
        return

    pairs = collect_pairs(hr_dir, lr_dir)
    n_total = len(pairs)
    if n_total == 0:
        print("❌ 找不到任何 HR/LR 成對檔案，請確認路徑與檔名是否對應。")
        return

    n_val = max(1, int(n_total * val_ratio))
    print(f"📊 總共有 {n_total} 組成對影像，準備切出 {n_val} 組作為驗證集 (val_ratio={val_ratio:.2f})")

    random.seed(seed)
    random.shuffle(pairs)

    val_pairs = pairs[:n_val]

    # 開始移動檔案
    for hr_path, lr_path in val_pairs:
        fname = os.path.basename(hr_path)

        new_hr_path = os.path.join(val_hr_dir, fname)
        new_lr_path = os.path.join(val_lr_dir, fname)

        print(f"  -> 移動 {fname} 到 val_hr / val_lr")
        shutil.move(hr_path, new_hr_path)
        shutil.move(lr_path, new_lr_path)

    print("✅ 分割完成！")
    print(f"   訓練集 HR 目錄: {hr_dir}")
    print(f"   驗證集 HR 目錄: {val_hr_dir}")
    print(f"   訓練集 LR 目錄: {lr_dir}")
    print(f"   驗證集 LR 目錄: {val_lr_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="將 data/train_hr & data/train_lr 中的一部分影像，移動到 val_hr / val_lr 做驗證集。"
    )
    parser.add_argument(
        "--hr-dir",
        type=str,
        default="data/train_hr",
        help="訓練用 HR 資料夾路徑（預設: data/train_hr）",
    )
    parser.add_argument(
        "--lr-dir",
        type=str,
        default="data/train_lr",
        help="訓練用 LR 資料夾路徑（預設: data/train_lr）",
    )
    parser.add_argument(
        "--val-hr-dir",
        type=str,
        default="data/val_hr",
        help="驗證用 HR 資料夾路徑（預設: data/val_hr）",
    )
    parser.add_argument(
        "--val-lr-dir",
        type=str,
        default="data/val_lr",
        help="驗證用 LR 資料夾路徑（預設: data/val_lr）",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="驗證集比例（預設 0.1 = 10%%）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="隨機種子，確保每次切分一致（預設 42）",
    )

    args = parser.parse_args()

    split_train_val(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        val_hr_dir=args.val_hr_dir,
        val_lr_dir=args.val_lr_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()