import numpy as np

# npzファイルをロード
file_path = '/home/h-okano/DiffBinaural/processed_data/3d_mapping_npz/000003.mp4/000002.npz'
data = np.load(file_path)

# npzファイルの中身を確認（キーを表示）
print("Keys in the npz file:", data.files)

# 各キーに対応する配列の内容を表示
for key in data.files:
    print(f"Contents of {key}:")
    print(data[key])
