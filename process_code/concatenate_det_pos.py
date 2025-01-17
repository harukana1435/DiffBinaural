import numpy as np
import os
# ファイルパス
# 対象ディレクトリのパス
det_dir = '/home/h-okano/DiffBinaural/processed_data/det_npy'
pointcloud_dir = 'DiffBinaural/processed_data/pointcloud'
output_dir = '/home/h-okano/DiffBinaural/processed_data/det_pos_npy'

# .npy と .npz の読み込み
npy_data = np.load(npy_file_path, allow_pickle=True).item()  # 辞書形式で扱えるように読み込む
npz_data = np.load(npz_file_path)  # npzデータの読み込み (H, W, 3)

# 必要なキーの確認
bounding_boxes = npy_data['bounding_boxes']  # バウンディングボックスのデータ
xyz_coords = npz_data['depth_map_3d']  # npzデータの座標 (H, W, 3)
print(bounding_boxes.shape)

# 3D位置データを格納するリスト
pos_3d = []

num_frame, num_sources, bb = bounding_boxes.shape

for frame in range(num_frame):
    
    for source in range(num_sources):
        x_min, y_min, x_max, y_max = bounding_boxes[frame, source]
        # 中心座標 (x_center, y_center)
        x_center = (x_min + x_max) // 2
        y_center = (y_min + y_max) // 2

        # npz_data の中心位置の座標を取得 (zはそのまま利用)
        pos = xyz_coords[y_center, x_center]
        pos_3d.append(pos)
        

# npy_data に '3d_pos' を追加
npy_data['pos_3d'] = np.array(pos_3d, dtype=float)

# 結果を確認
print("3D positions added to npy_data['3d_pos']:")
#print(npy_data['pos_3d'])

# 必要に応じて npy ファイルを保存
#output_path = '/home/h-okano/DiffBinaural/processed_data/det_npy/000001_updated.npy'
#np.save(output_path, npy_data)
#print(f"\nUpdated npy data saved to: {output_path}")

def concatenate_det_pos(det_path,pointcloud_path, output_path):
    # .npy と .npz の読み込み
    npy_data = np.load(det_path, allow_pickle=True).item()  # 辞書形式で扱えるように読み込む
    npz_data = np.load(pointcloud_path)  # npzデータの読み込み (H, W, 3)

    bounding_boxes = npy_data['bounding_boxes']  # バウンディングボックスのデータ
    xyz_coords = npz_data['depth_map_3d']  # npzデータの座標 (H, W, 3)
    # 3D位置データを格納するリスト
    pos_3d = []

    num_frame, num_sources, bb = bounding_boxes.shape

    for frame in range(num_frame):
        frame_id=(num_frame+1)*2
        pointcloud_filepath = os.path.join(pointcloud_path, f"{frame_id:06}", ".npz")
        for source in range(num_sources):
            x_min, y_min, x_max, y_max = bounding_boxes[frame, source]
            # 中心座標 (x_center, y_center)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # npz_data の中心位置の座標を取得 (zはそのまま利用)
            pos = xyz_coords[y_center, x_center]
            pos_3d.append(pos)



if __name__=="__main__":
    for filename in os.listdir(det_dir):
        if filename.endswith('.npy'):
            det_path = os.path.join(det_dir,filename)
            basename = os.path.splitext(filename)
            pointcloud_path = os.path.join(pointcloud_dir, basename, '.mp4')
            output_path = os.path.join(output_dir, basename, '.npy')
            concatenate_det_pos(det_path,pointcloud_path, output_path)
            