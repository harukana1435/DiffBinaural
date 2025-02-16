import os
import numpy as np

# ディレクトリのパス
directory_path = "/home/h-okano/DiffBinaural/processed_data/det_pos_npy"

def count_det_num(directory):
    max = 0
    num_1 = 0
    num_2 = 0
    num_3 = 0
    num_4 = 0
    # 指定されたディレクトリ内のすべてのファイルを走査
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            print(f"File: {filename}")
            
            try:
                # .npyファイルを読み込む
                data = np.load(file_path, allow_pickle=True).item()
                bounding_box_shape = data['bounding_boxes'].shape[1]
                pos_3d_shape = data['pos_3d'].shape[1]
                if(bounding_box_shape == pos_3d_shape):
                    print(bounding_box_shape)
                    if(max<bounding_box_shape):
                        max = bounding_box_shape
                else:
                    print("what the")
                
                if bounding_box_shape == 1:
                    num_1 += 1
                elif bounding_box_shape == 2:
                    num_2 += 1
                elif bounding_box_shape == 3:
                    num_3 +=1 
                elif bounding_box_shape == 4:
                    num_4 += 1
                
                print(f"バウンディングボックス{bounding_box_shape} 位置{pos_3d_shape}")
                # 辞書形式の場合キーを表示
                if isinstance(data, dict):
                    print("  Keys:", list(data.keys()))
                else:
                    print("  This .npy file is not a dictionary.")
            except Exception as e:
                print(f"  Error reading file {filename}: {e}")
            print("-" * 50)
    print(f"最大値：{max} 1つ {num_1} 2つ {num_2} 3つ {num_3} 4つ {num_4}")

# 実行
count_det_num(directory_path)
