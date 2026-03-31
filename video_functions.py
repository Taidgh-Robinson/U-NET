import cv2
import numpy as np
import torch
from torchvision import transforms
from helper_functions import apply_model_to_whole_image
from PIL import Image


def extract_frames(video_path, every_n_frames=1):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_idx += 1

    cap.release()
    return frames


def convert_frame_to_model_input(frame):
    process = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )
    return process(frame)


def apply_model_to_entire_video(model, video_path):
    i = 0
    frames = extract_frames(video_path)
    for frame in frames:
        processed = convert_frame_to_model_input(frame)
        output = apply_model_to_whole_image(model, processed)
        output = output.permute(1, 2, 0).unsqueeze(-1)
        actual_output = output.argmax(dim=2)
        mask = actual_output.squeeze().numpy()  # [H, W]
        masked_frame = frame.copy()
        masked_frame[mask == 0] = 0  # zero out pixels model thinks are background
        img = Image.fromarray(masked_frame)
        img.save(f"movie_output/frame_{i:04d}.png")
        i += 1
