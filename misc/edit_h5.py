import h5py  # HDF5のライブラリ
import os

# 読み出し
filename = "./splits/split1/train.h5"  # ここに読み込みたいファイル名
h5file = h5py.File(filename, "r+")

# HDF5ファイル全体の内容を見たい場合
def print_hdf5_structure(group, indent=0):
    for key in group:
        item = group[key]
        print("  " * indent + key)
        if isinstance(item, h5py.Group):
            print_hdf5_structure(item, indent + 1)
        elif isinstance(item, h5py.Dataset):
            print("  " * (indent + 1) + str(item.shape))
            print("  " * (indent + 1) + str(item[:]))


# 変更したい文字列の置換関数
def replace_paths_in_hdf5(group):
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            replace_paths_in_hdf5(item)
        elif isinstance(item, h5py.Dataset):
            data = item[:]
            if data.dtype.kind == 'S':  # バイト文字列の場合
                new_data = [s.replace(
                    b'/home/h-okano/DETbinaural/dataset/binaural_audios/', 
                    b'/home/h-okano/DETbinaural/dataset/binaural_audios_16000Hz/') for s in data]
                item[...] = new_data

# 指定ディレクトリ内のすべてのHDF5ファイルに置換を適用する関数
def replace_paths_in_all_hdf5_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".h5"):
                file_path = os.path.join(root, file)
                with h5py.File(file_path, "r+") as h5file:
                    replace_paths_in_hdf5(h5file)
                print(f"Replaced paths in {file_path}")

# 処理するディレクトリのパス
base_directory = "./splits"

# すべてのHDF5ファイルに置換を適用
#replace_paths_in_all_hdf5_files(base_directory)

# HDF5ファイル内のパスを置換
#replace_paths_in_hdf5(h5file)

# HDF5ファイルの構造を表示
print_hdf5_structure(h5file)

