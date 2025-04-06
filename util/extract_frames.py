import os
import time
import cv2

# Video file to process
video_file = "./data/nb.mp4"
output_dir = "./trash"

# Folders in the output directory
# final - list of unique pages in the video file
# frames - all frames in the video file
# pages - pages in all frames
for d in ["final", "frames", "pages"]:
    os.makedirs(output_dir + "/" + d, exist_ok=True)

def video_to_frames(video_file):
    vidcap = cv2.VideoCapture(video_file)

    frames = []

    while True:
        success, image = vidcap.read()
        if success:
            frames.append(image)
        else:
            break

    return frames

def save_images(images, output_dir):
    for i, f in enumerate(images):
        cv2.imwrite(f"{output_dir}/{i:05d}.jpg", f)

frames = video_to_frames(video_file)
save_images(frames, output_dir + "/frames")
