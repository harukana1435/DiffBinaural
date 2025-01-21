import os
import numpy as np
import cv2
import open3d as o3d

def extract_combined_pointcloud_to_ply(image_path, pointcloud_path, bbox_path, output_ply_path):
    """
    画像と3次元座標データを基に、すべてのバウンディングボックス領域を切り取り、
    統合した点群データを生成し、PLY形式で保存します。

    Args:
        image_path (str): 入力画像のパス。
        pointcloud_path (str): 入力3次元座標データのパス。
        bbox_path (str): バウンディングボックス情報のファイルパス (.npy)。
        output_ply_path (str): 点群データの出力パス (.ply)。
    """
    # データ読み込み
    image = cv2.imread(image_path)  # (H, W, 3) の画像
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pointcloud = np.load(pointcloud_path)['depth_map_3d']  # (H, W, 3) の3次元座標
    bbox_data = np.load(bbox_path, allow_pickle=True).item()
    print(bbox_data)

    bounding_boxes = bbox_data['bounding_boxes']  # バウンディングボックス (N, 4)
    labels = bbox_data['labels']                 # ラベル情報 (N,)
    scores = bbox_data['scores']                 # スコア (N,)
    
    bounding_boxes = [[0,0,1280,720]]

    combined_points = []  # 統合された点群データ
    combined_colors = []  # 統合された色情報

    # 各バウンディングボックスで処理
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = bbox

        # バウンディングボックス領域を切り取り
        cropped_image = image[y1:y2, x1:x2]                # 切り取った画像
        cropped_pointcloud = pointcloud[y1:y2, x1:x2, :]  # 切り取った3次元座標

        # 点群データ (X, Y, Z)
        H, W, _ = cropped_image.shape
        y, x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        points = cropped_pointcloud.reshape(-1, 3)

        # 色データ (R, G, B)
        colors = cropped_image.reshape(-1, 3) / 255.0  # 0-1 に正規化

        # 統合
        combined_points.append(points)
        combined_colors.append(colors)

    # 統合データを1つの配列にまとめる
    combined_points = np.vstack(combined_points)
    combined_colors = np.vstack(combined_colors)

    # Open3D 点群オブジェクトを作成
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_points)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    # 点群の座標系を調整
    pcd.transform([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # PLYファイルに保存
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"Point cloud saved to {output_ply_path}")

# 実行例
if __name__ == "__main__":
    image_path = "/home/h-okano/DiffBinaural/FairPlay/frames/000014.mp4/000060.jpg"
    pointcloud_path = "/home/h-okano/DiffBinaural/processed_data/pointcloud/000014.mp4/000060.npz"
    bbox_path = "/home/h-okano/DiffBinaural/processed_data/det_npy/000014.npy"
    output_ply_path = "/home/h-okano/DiffBinaural/misc/temp/combined_pointcloud.ply"

    extract_combined_pointcloud_to_ply(image_path, pointcloud_path, bbox_path, output_ply_path)
