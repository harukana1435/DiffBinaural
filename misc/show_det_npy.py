import numpy as np

# ファイルパスを指定
file_path = '/home/h-okano/DiffBinaural/processed_data/det_npy/000887.npy'

# .npy ファイルを読み込む
try:
    # allow_pickle=True を設定
    data = np.load(file_path, allow_pickle=True)
    print("データの内容:")
    print(data)  # 配列の内容を表示
    print("\nデータの情報:")
    print(f"型: {type(data)}")  # データ型
    print(f"形状: {data.shape if isinstance(data, np.ndarray) else '不明'}")  # 配列の形状
    print(f"データ型: {data.dtype if isinstance(data, np.ndarray) else '不明'}")  # 配列のデータ型
except Exception as e:
    print(f"エラーが発生しました: {e}")
