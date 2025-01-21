import numpy as np
import os
# ファイルパス
# 対象ディレクトリのパス
det_dir = '/home/h-okano/DiffBinaural/processed_data/det_npy'
pointcloud_dir = '/home/h-okano/DiffBinaural/processed_data/pointcloud'
output_dir = '/home/h-okano/DiffBinaural/processed_data/det_pos_npy'




def concatenate_det_pos(det_path,pointcloud_path, output_path):
    # .npy と .npz の読み込み
    npy_data = np.load(det_path, allow_pickle=True).item()  # 辞書形式で扱えるように読み込む
    

    bounding_boxes = npy_data['bounding_boxes']  # バウンディングボックスのデータ
    # 3D位置データを格納するリスト
    pos_3d = []

    if(bounding_boxes.ndim == 2):
        bounding_boxes = np.tile(bounding_boxes, (40, 1, 1))
    num_frame, num_sources, bb = bounding_boxes.shape

    for frame in range(num_frame):
        pos_3d_source = []
        frame_id=(frame+1)*2
        pointcloud_filepath = os.path.join(pointcloud_path, f"{frame_id:06}.npz")
        #print(frame_id)
        npz_data = np.load(pointcloud_filepath)  # npzデータの読み込み (H, W, 3)
        xyz_coords = npz_data['depth_map_3d']  # npzデータの座標 (H, W, 3)
        
        for source in range(num_sources):
            x_min, y_min, x_max, y_max = bounding_boxes[frame, source]
            # 中心座標 (x_center, y_center)
            x_center = (x_min + x_max) // 2
            y_center = (y_min + y_max) // 2

            # npz_data の中心位置の座標を取得 (zはそのまま利用)
            pos = xyz_coords[y_center, x_center]
            x, y, z = pos
            # r の計算 (原点からの距離)
            r = np.sqrt(x**2 + y**2 + z**2)

            # θ (仰角) の計算
            theta = np.arctan2(y, np.sqrt(x**2+z**2))

            # φ (方位角) の計算
            phi = np.arctan2(x, z)
            
            pos_3d_source.append([r, np.degrees(theta), np.degrees(phi)])
        pos_3d.append(pos_3d_source)
        
    pos_3d = np.array(pos_3d, dtype=float)
    #print(pos_3d)
    # npy_data に '3d_pos' を追加
    npy_data['pos_3d'] = np.array(pos_3d, dtype=np.float16)
    
    np.save(output_path, npy_data)



if __name__=="__main__":
    for filename in os.listdir(det_dir):
        if filename.endswith('.npy'):
            det_path = os.path.join(det_dir,filename)
            basename = os.path.splitext(filename)[0]
            print(basename)
            pointcloud_path = os.path.join(pointcloud_dir, basename+'.mp4')
            output_path = os.path.join(output_dir, basename+'.npy')
            if os.path.exists(output_path):
                print(f"continue {basename}")
                continue
            concatenate_det_pos(det_path,pointcloud_path, output_path)
        
            