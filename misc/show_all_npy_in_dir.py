import os
import numpy as np
import argparse

# matplotlibは必要な場合のみインポート
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

def show_npy_info(file_path, summary_dict):
    try:
        data = np.load(file_path, allow_pickle=True)
        if data is None:
            print(f"{file_path}: None")
            summary_dict['None'] = summary_dict.get('None', 0) + 1
        elif isinstance(data, np.ndarray):
            if data.shape == ():
                print(f"{file_path}: shape = (), value = {data}")
                summary_dict['ndarray_shape_()'] = summary_dict.get('ndarray_shape_()', 0) + 1
            elif data.size == 0:
                print(f"{file_path}: empty array, shape = {data.shape}")
                summary_dict[f'ndarray_empty_shape_{data.shape}'] = summary_dict.get(f'ndarray_empty_shape_{data.shape}', 0) + 1
            else:
                print(f"{file_path}: shape = {data.shape}")
                summary_dict[f'ndarray_shape_{data.shape}'] = summary_dict.get(f'ndarray_shape_{data.shape}', 0) + 1
        elif isinstance(data, dict):
            print(f"{file_path}: dict with keys: {list(data.keys())}")
            dict_shape_summary = {}
            for k, v in data.items():
                if hasattr(v, 'shape'):
                    print(f"  key '{k}': shape = {v.shape}")
                    dict_shape_summary[k] = f'shape_{v.shape}'
                elif hasattr(v, '__len__'):
                    print(f"  key '{k}': len = {len(v)}")
                    dict_shape_summary[k] = f'len_{len(v)}'
                else:
                    print(f"  key '{k}': type = {type(v)}")
                    dict_shape_summary[k] = f'type_{type(v)}'
            dict_shape_summary_str = str(sorted(dict_shape_summary.items()))
            summary_dict[f'dict_{dict_shape_summary_str}'] = summary_dict.get(f'dict_{dict_shape_summary_str}', 0) + 1
        else:
            print(f"{file_path}: type = {type(data)} (not ndarray or dict)")
            summary_dict[f'other_{type(data)}'] = summary_dict.get(f'other_{type(data)}', 0) + 1
    except Exception as e:
        print(f"  Error loading {file_path}: {e}")
        summary_dict['error'] = summary_dict.get('error', 0) + 1

def main():
    parser = argparse.ArgumentParser(description="Show info for all .npy files in a directory and summarize shapes/types.")
    parser.add_argument("dir", type=str, help="Directory containing .npy files")
    args = parser.parse_args()

    npy_files = [f for f in os.listdir(args.dir) if f.endswith('.npy')]
    if not npy_files:
        print(f"No .npy files found in {args.dir}")
        return
    summary_dict = {}
    for fname in sorted(npy_files):
        show_npy_info(os.path.join(args.dir, fname), summary_dict)
    print("\n=== Summary of file types and shapes ===")
    for k, v in summary_dict.items():
        print(f"{k}: {v} files")

if __name__ == "__main__":
    main() 