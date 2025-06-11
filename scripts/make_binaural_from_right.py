import os
import numpy as np
from scipy.io import wavfile
import shutil

SRC_ROOT = 'DiffBinaural/generated_audios/results_right-sin'
DST_ROOT = 'DiffBinaural/generated_audios/results_right-binaural'

# サブディレクトリを列挙
dirnames = [d for d in os.listdir(SRC_ROOT) if os.path.isdir(os.path.join(SRC_ROOT, d))]
dirnames.sort()

for d in dirnames:
    src_dir = os.path.join(SRC_ROOT, d)
    dst_dir = os.path.join(DST_ROOT, d)
    input_path = os.path.join(src_dir, 'input_binaural.wav')
    mixed_path = os.path.join(src_dir, 'mixed_mono.wav')
    right_path = os.path.join(src_dir, 'predicted_binaural.wav')
    out_path = os.path.join(dst_dir, 'predicted_binaural.wav')

    # 必要なファイルが揃っているか
    if not (os.path.exists(input_path) and os.path.exists(mixed_path) and os.path.exists(right_path)):
        print(f"skip {d}: 必要なwavファイルが見つかりません")
        continue

    # 出力先ディレクトリ作成
    os.makedirs(dst_dir, exist_ok=True)

    # wav読み込み
    sr1, mixed = wavfile.read(mixed_path)
    sr2, right = wavfile.read(right_path)
    if sr1 != sr2:
        print(f"skip {d}: sample rate mismatch")
        continue

    # shape調整
    mixed = mixed.squeeze()
    right = right.squeeze()

    # rightが2chの場合は右chのみ抽出
    if right.ndim == 2:
        # 右ch（1ch目 or 2ch目）を選択（通常は2ch目）
        right = right[:, -1]
    if mixed.ndim == 2:
        # mixedも1ch化（通常はモノラルなので0ch目）
        mixed = mixed[:, 0]

    min_len = min(len(mixed), len(right))
    mixed = mixed[:min_len]
    right = right[:min_len]

    # 左ch = mixed_mono - 右ch
    left = mixed.astype(np.int32) - right.astype(np.int32)
    left = np.clip(left, -32768, 32767).astype(np.int16)
    binaural = np.stack([left, right], axis=-1)

    # 保存
    wavfile.write(out_path, sr1, binaural)
    print(f"wrote {out_path}")

    # mixed_mono.wav, input_binaural.wavもコピー
    shutil.copy2(mixed_path, os.path.join(dst_dir, 'mixed_mono.wav'))
    shutil.copy2(input_path, os.path.join(dst_dir, 'input_binaural.wav')) 