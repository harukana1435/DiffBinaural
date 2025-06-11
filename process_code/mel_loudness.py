import os
import torch

def calculate_average_loudness_difference(left_dir, right_dir):
    """
    Calculates the average loudness difference between left and right channel mel spectrograms.
    
    Args:
        left_dir: Path to the directory containing left channel mel spectrograms (saved with torch.save).
        right_dir: Path to the directory containing right channel mel spectrograms (saved with torch.save).

    Returns:
        The average loudness difference. Returns None if no files are found or an error occurs.
    """
    left_files = [f for f in os.listdir(left_dir) if f.endswith('.npy')]
    right_files = [f for f in os.listdir(right_dir) if f.endswith('.npy')]

    if not left_files or not right_files:
        print("Error: No .npy files found in either directory.")
        return None

    if len(left_files) != len(right_files):
        print("Error: Number of files in left and right directories do not match.")
        return None

    loudness_diffs = []
    loudness_lefts = []
    loudness_rights = []
    for file_name in left_files:
        left_path = os.path.join(left_dir, file_name)
        right_path = os.path.join(right_dir, file_name)

        try:
            left_mel = torch.load(left_path)
            right_mel = torch.load(right_path)

            # Calculate loudness difference using mean absolute difference
            loudness_left = torch.sqrt(torch.mean(torch.pow(left_mel[:, :-1], 2))).item()
            loudness_lefts.append(loudness_left)
            loudness_right = torch.sqrt(torch.mean(torch.pow(right_mel[:, :-1], 2))).item()
            loudness_rights.append(loudness_right)
            loudness_diff = abs(loudness_left - loudness_right)
            loudness_diffs.append(loudness_diff)

        except FileNotFoundError:
            print(f"Error: File not found: {file_name}")
            return None
        except Exception as e:
            print(f"An error occurred while processing {file_name}: {e}")
            return None

    average_loudness_diff = sum(loudness_diffs) / len(loudness_diffs) if loudness_diffs else None
    average_loudness_left = sum(loudness_lefts) / len(loudness_lefts) if loudness_lefts else None
    average_loudness_right = sum(loudness_rights) / len(loudness_rights) if loudness_rights else None
    return average_loudness_diff, average_loudness_left, average_loudness_right

if __name__ == "__main__":
    left_directory = "/home/h-okano/DiffBinaural/processed_data/normalized_mel_left"
    right_directory = "/home/h-okano/DiffBinaural/processed_data/generated_mel_right_2ch"

    average_diff, average_loudness_left, average_loudness_right = calculate_average_loudness_difference(left_directory, right_directory)

    if average_diff is not None:
        print(f"Average loudness difference: {average_diff}")
        print(f"Average loudness left: {average_loudness_left}")
        print(f"Average loudness right: {average_loudness_right}")
