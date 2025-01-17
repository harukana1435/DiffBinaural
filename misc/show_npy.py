import numpy as np
import os
import matplotlib.pyplot as plt

def display_npy_file(npy_file_path):
    """.npyファイルを表示する関数"""
    if not os.path.exists(npy_file_path):
        print(f"File not found: {npy_file_path}")
        return

    # .npyファイルをロード
    data = np.load(npy_file_path, allow_pickle=True)
    print(data)
    print(f"Loaded .npy file: {npy_file_path}")
    print(f"Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")
    print(f"Min Value: {data.min()}, Max Value: {data.max()}")
    
    # データを可視化（3Dデータが画像の場合）
    if len(data.shape) == 2 or (len(data.shape) == 3 and data.shape[2] in [1, 3, 4]):
        plt.imshow(data if data.shape[-1] != 1 else data[..., 0], cmap='viridis')
        plt.title(f"Visualization of {npy_file_path}")
        plt.colorbar()
        plt.show()
    else:
        print("Data visualization not supported for this shape.")

# サンプル使用方法
if __name__ == "__main__":
    # 表示したい .npy ファイルのパス
    npy_file_path = "/home/h-okano/DiffBinaural/processed_data/det_npy/001629.npy"
    display_npy_file(npy_file_path)
