import h5py

def explore_h5(file_path):
    """HDF5ファイルの内容を表示"""
    def print_attrs(name, obj):
        """グループやデータセットの属性を表示"""
        print(f"Name: {name}")
        if isinstance(obj, h5py.Dataset):
            print(f"  Type: Dataset")
            print(f"  Shape: {obj.shape}")
            print(f"  Data type: {obj.dtype}")
            print(f"  Data (first 5 elements): {obj[:5]}")
        elif isinstance(obj, h5py.Group):
            print(f"  Type: Group")
        else:
            print(f"  Type: Unknown")
        for key, value in obj.attrs.items():
            print(f"  Attribute: {key} => {value}")
        print()
    
    # HDF5ファイルを開く
    with h5py.File(file_path, 'r') as f:
        print(f"Exploring file: {file_path}")
        # 再帰的に内容を表示
        f.visititems(print_attrs)

# 使用例
h5_file_path = "/home/h-okano/Diffbinaural/FairPlay/splits/split1/train.h5"  # 自分のHDF5ファイルに変更
explore_h5(h5_file_path)
