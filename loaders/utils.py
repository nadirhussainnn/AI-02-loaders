import librosa
import cv2
import torch
import numpy as np

def process_video(video_path, transform):
    print(f"Processing video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open {video_path}")
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:  # End of video
            break

        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert frame to tensor
        frame_tensor = transform(frame_rgb)
        frames.append(frame_tensor)

    cap.release()

    # Stack frames into a single tensor
    video_tensor = torch.stack(frames)  # Shape: (num_frames, C, H, W)
    print(f"Processed video shape: {video_tensor.shape}")
    return video_tensor

def process_audio(file_path):
    # Load audio with librosa
    audio, sr = librosa.load(file_path, sr=None)  # sr=None preserves original sampling rate
    print(f"Loaded {file_path}, Sampling Rate: {sr}, Audio Shape: {audio.shape}")

    # Convert to mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)

    # Convert mel spectrogram to log scale
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Normalize to range [0, 1]
    normalized_spectrogram = (log_mel_spectrogram - log_mel_spectrogram.min()) / (
        log_mel_spectrogram.max() - log_mel_spectrogram.min()
    )

    # Convert to a tensor
    spectrogram_tensor = torch.tensor(normalized_spectrogram, dtype=torch.float32)

    return spectrogram_tensor, sr

