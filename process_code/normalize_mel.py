import os
import numpy as np
import torch

def calculate_rms(mel_spectrogram):
    return np.sqrt(np.mean(np.square(mel_spectrogram)))

def normalize_mel(left_mel_dir, right_mel_dir):
    left_rms = 0.1365
    right_rms = 0.1496
    
    # Get the list of files in the left and right mel directories
    left_files = sorted(os.listdir(left_mel_dir))
    right_files = sorted(os.listdir(right_mel_dir))

    # Check if the number of files in the left and right directories are the same
    if len(left_files) != len(right_files):
        raise ValueError("The number of files in the left and right directories must be the same.")

    # Iterate over the files in the left and right mel directories
    for left_file, right_file in zip(left_files, right_files):
        # Load the left and right mel spectrograms
        left_mel_path = os.path.join(left_mel_dir, left_file)
        right_mel_path = os.path.join(right_mel_dir, right_file)

        left_mel = torch.load(left_mel_path)
        right_mel = torch.load(right_mel_path)

        # Calculate the scaling factor
        scaling_factor = right_rms /left_rms 

        # Apply the scaling factor to the left channel mel spectrogram
        normalized_left_mel = left_mel * scaling_factor

        # Save the normalized left mel spectrogram
        output_dir = "/home/h-okano/DiffBinaural/processed_data/normalized_mel_left"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, left_file)
        torch.save(normalized_left_mel, output_path)

if __name__ == "__main__":
    left_mel_dir = "/home/h-okano/DiffBinaural/processed_data/generated_mel_left_2ch"
    right_mel_dir = "/home/h-okano/DiffBinaural/processed_data/generated_mel_right_2ch"
    normalize_mel(left_mel_dir, right_mel_dir)
