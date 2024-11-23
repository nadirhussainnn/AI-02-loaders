import pandas as pd
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from utils import *

audio_folder = "./mnist-audios"
video_folder = "./videos"

def load_csv():
    df = pd.read_csv("./csvs/iris_dataset.csv")
    print(df.info())


def load_github_csv():
    df = pd.read_csv("https://raw.githubusercontent.com/fpinell/mlsa/refs/heads/main/data/iris_dataset.csv")
    print(df.head())
    
def load_images():
    
    image = Image.open("./images/bike_001.bmp")
    transform = transforms.ToTensor()
    image_tensor = transform(image)

    # PyTorch handles tensors as (c, h, w) , while matplotlib needs as (h, w, c) so let's convert
    converted_tensor = image_tensor.permute(1,2,0)
    plt.imshow(converted_tensor)
    plt.axis("off")
    plt.show()

def load_videos():

    # Transformation: Convert frames to tensors
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts (H, W, C) -> (C, H, W)
    ])

    # Iterate through the folder
    all_videos_tensors = {}
    for filename in os.listdir(video_folder):
        if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Supported video formats
            video_path = os.path.join(video_folder, filename)
            video_tensor = process_video(video_path, transform)

            if video_tensor is not None:
                all_videos_tensors[filename] = video_tensor

    if all_videos_tensors:
        first_video = list(all_videos_tensors.keys())[0]
        first_video_tensor = all_videos_tensors[first_video]
        plt.imshow(first_video_tensor[0].permute(1, 2, 0))  # (H, W, C)
        plt.axis("off")
        plt.title(f"First Frame: {first_video}")
        plt.show()

def load_audios():
    audio_data = {}
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):  # Assuming MNIST-Audio contains .wav files
            file_path = os.path.join(audio_folder, filename)
            spectrogram_tensor, sr = process_audio(file_path)
            audio_data[filename] = spectrogram_tensor

    if audio_data:
        first_file = list(audio_data.keys())[0]
        first_spectrogram = audio_data[first_file]

        plt.figure(figsize=(10, 4))
        plt.imshow(first_spectrogram.numpy(), aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Log Mel Spectrogram: {first_file}")
        plt.xlabel("Time Frames")
        plt.ylabel("Mel Frequency Bands")
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    load_csv()
    load_github_csv()
    load_images()
    load_videos()
    load_audios()