import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from PIL import Image

start_time = start_time = time.perf_counter()

# Video file to process
# video_file = "./data/nb.mp4"
video_file = input("Enter Video Location: ") or "./data/nb.mp4"
output_dir = "./notebooks/trash"
print("Processing", video_file, "...")

# Folders in the output directory
# final - list of unique pages in the video file
# frames - all frames in the video file
# pages - pages in all frames
for d in ["final", "frames", "pages", "result"]:
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

print(f"No of frames {len(frames)}")

import sys
# sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import segment_anything
print(f"Imported segment_anything from {segment_anything.__file__}")

sam_checkpoint = "notebooks/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# Filter out the masks to get the list of potential page(s)
# Return of a list of masks that are potential pages
# Consider the masks whose area > apct * frame_size
# Accept a mask that has < X (area * fpct) number of False inside the bbox
def get_pages(masks, apct, fpct):
    p = []
    if len(masks) == 0:
        return p

    frame_size = masks[0]['segmentation'].size

    for m in masks:
        if m['area'] > frame_size * apct:
            bbox = m['bbox']
            num_falses = np.count_nonzero(m['segmentation'][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] == False)
            if num_falses < m['area'] * fpct:
                p.append(m)
    return p

# If two pages are in a frame, then there maybe 3 segments
# 1. covering both pages
# 2. page 1
# 3. page 2
# Filter out the page #1, since the bbox is covered by #2 + #3
def filter_covered_pages(pages, tpct):
    filtered = []
    if len(pages) == 0:
        return filtered

    img_mask = np.full(pages[0]['segmentation'].shape, False)

    for p in sorted(pages, key=(lambda x: x['area'])):
        bbox = p['bbox']
        num_trues = np.count_nonzero(img_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] == True)
        if num_trues < p['area'] * tpct:
            filtered.append(p)
        img_mask[p['segmentation']] = True
    return filtered

pgs = []
fl = []

for index, fr in enumerate(frames):
    t1 = time.monotonic_ns()
    masks = mask_generator.generate(fr)
    t2 = time.monotonic_ns()
    print(f"Frame number {index} - time taken in generating masks {(t2 - t1) / 1000000} milli seconds")
    masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
    print(f"No of masks {len(masks)}")

    pgs.append(get_pages(masks, 0.20, 0.20))
    fl.append(filter_covered_pages(get_pages(masks, 0.20, 0.20), 0.20))

def mono(fr, tr, tl):
    frs = fr['segmentation']
    for n4 in range(0, 4):
        ft = []
        for i in frs:
            x = 0
            for j in i:
                if j == True:
                    break
                x += 1
            ft.append(x)
    
        fi = False
        for i in range(0, len(ft)-tl):
            if abs(ft[i] - ft[i+tl])*tr > tl:
                if ft[i+tl] > ft[i]:
                    fi = True
                elif fi:
                    return False
        frs = np.rot90(frs)
    
    return True

goodpg = []
for i in fl:
    g = (len(i) == 2)
    for j in i:
        g = (g and mono(j, 2, 10))
    goodpg.append(g)

x = 0
for i in goodpg:
    print(i, x)
    x += 1

def final_frames(gp):
    tf = [-1]
    for i in range(0, len(gp)-1):
        if gp[i] != gp[i+1]:
            tf.append(i)
    if len(tf) % 2 == 1:
        tf.append(len(gp)-1)
    ff = []
    for i in range(0, len(tf)-1, 2):
        ff.append((tf[i+1] + tf[i]+1)//2)
    return ff

ff = final_frames(goodpg)
for i in range(0, len(ff)):
    print("FF", ff[i])
    im = Image.fromarray(frames[ff[i]])
    im.save(output_dir + "/result/pg" + str(i) + "frame" + str(ff[i]) + ".jpg")

from PIL import Image
import os

def convert_jpg_to_pdf(image_folder, output_pdf):
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg'))]
    if not image_files:
        print("No JPG images found in the specified folder.")
        return
    image_files.sort()

    image_paths = [os.path.join(image_folder, img) for img in image_files]
    images = [Image.open(path).convert('RGB') for path in image_paths]

    images[0].save(output_pdf, save_all=True, append_images=images[1:])
    print(f"Successfully created PDF: {output_pdf}")

image_folder = output_dir + "/result"
output_pdf = "output.pdf"
convert_jpg_to_pdf(image_folder, output_pdf)

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
