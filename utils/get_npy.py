import numpy as np

def load_and_view_npy(file_path):
    # npyファイルの読み込み
    data = np.load(file_path, allow_pickle=True)

    # 読み込んだデータを表示
    print("Loaded data:")
    print(data)
    
    # もし辞書形式で保存されている場合、その中身も表示
    if isinstance(data, dict):
        for key, value in data.items():
            print(f"{key}: {value}")

# 使用例
file_path = '../processed_data/det_npy/000583.npy'  # ここに確認したいnpyファイルのパスを指定
load_and_view_npy(file_path)
