import time
import numpy as np
import cv2
import sys
import torch
import os
from PIL import Image
from ultralytics import FastSAM

# Folders in the output directory
# final - list of unique pages in the video file
# frames - all frames in the video file
# pages - pages in all frames

def video_to_frames(video_file, fps=1):
    vidcap = cv2.VideoCapture(video_file)
    frames = []
    video_fps = vidcap.get(cv2.CAP_PROP_FPS)
    print(f"Video FPS: {video_fps}")
    if video_fps <= 0:
        video_fps = 30  # fallback if FPS not detected
    frame_interval = int(round(video_fps / fps))
    count = 0
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frames.append(image)
        count += 1
    vidcap.release()
    return frames

def save_images(images, output_dir):
    for i, f in enumerate(images):
        cv2.imwrite(f"{output_dir}/{i:05d}.jpg", f)

def find_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None  # Handle empty mask case

    min_row, max_row = np.where(rows)[0][[0, -1]]
    min_col, max_col = np.where(cols)[0][[0, -1]]
    #return min_row, max_row, min_col, max_col
    return min_col, min_row, max_col-min_col, max_row-min_row

def genfs_numpy(image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eres = model(image, device=device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9, verbose = False)
    masks_cpu = eres[0].masks.data.cpu().detach().numpy()
    xyxy = eres[0].boxes.xyxy.cpu().detach().numpy().astype(int)

    ret = []
    for i in range(0, len(masks_cpu)):
        ret.append({
            'segmentation': masks_cpu[i],
            #'bbox': [xyxy[i][0], xyxy[i][1], xyxy[i][2]-xyxy[i][0], xyxy[i][3]-xyxy[i][1]],
            'bbox': find_bounding_box(masks_cpu[i]),
            'xyxy': xyxy[i],
            'area': np.count_nonzero(masks_cpu[i] == 1), # torch.sum(masks_cpu[i] == 1) when using tensors
        })

    return sorted(ret, key=(lambda x: x['area']), reverse=True)

# If two pages are in a frame, then there maybe 3 segments
# 1. covering both pages
# 2. page 1
# 3. page 2
# Filter out the page #1, since the bbox is covered by #2 + #3
def filter_covered_pages(pages, tpct): # *
    filtered = []
    if len(pages) == 0:
        return filtered
    elif len(pages) <= 2:
        return pages


    img_mask = np.full(pages[0]['segmentation'].shape, False)

    for p in sorted(pages, key=(lambda x: x['area'])):
        bbox = p['bbox']
        num_trues = np.count_nonzero(img_mask[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] == True)
        if num_trues < p['area'] * tpct:
            filtered.append(p)
        img_mask[p['segmentation'] == 1] = True
    return filtered

# uses upper and lower size bound
def find_pages_nb(masks, apct, amx, fpct): # *
    p = []
    if len(masks) == 0:
        return p

    frame_size = masks[0]['segmentation'].size
    for m in masks:
        if m['area'] > frame_size * apct and m['area'] < frame_size * amx:
            bbox = m['bbox']
            num_falses = np.count_nonzero(m['segmentation'][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]] == False)
            if num_falses < m['area'] * fpct:
                p.append(m)
    return p

def progress_bar(iteration, total, length=30, fill='█'): # *
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| frame {iteration} of {total} completed')
    sys.stdout.flush()

# Find if the pages are mono increasing then mono decreasing to check if anything is blocking a page
# slope over range is (slope) <-- filter for noise from sam

# OPTMIZE W NUMPY FUNCS
def mono(mask, rnge, slope): # *
    frs = mask['segmentation']
    for _ in range(0, 4):
        ft = []
        for i in frs:
            first1 = np.argmax(i)
            first1 = first1 if i[first1] != 0 else len(i)
            ft.append(first1)

        fi = False
        for i in range(0, len(ft)-rnge):
            if abs(ft[i] - ft[i+rnge])/rnge > slope:
                if ft[i+rnge] > ft[i]:
                    fi = True
                elif fi:
                    return False
        frs = np.rot90(frs)

    return True

# find the ranges of visible pages which will be the range in the video where a
# page is fully visible. Choose middle frame for final picture
def final_frames(gp): # *
    ff = []
    i = 0
    while i < len(gp):
        while i < len(gp) and not gp[i]:
            i += 1
        if i >= len(gp) or not gp[i]:
            break
        j = i
        while i+1 < len(gp) and gp[i+1]:
            i += 1
        i += 1
        ff.append((i+j)//2)
    return ff

def frames_to_pdf(frames, fin_pg, ff, output_pdf): # *
    pdf_images = []
    for pgi in ff:
        fin_pg[pgi] = sorted(fin_pg[pgi], key=lambda x:x['bbox'][0])
        for i in range(len(fin_pg[pgi])):
            bbox = fin_pg[pgi][i]['bbox']
            cropped = frames[pgi][bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
            im = Image.fromarray(cropped)
            pdf_images.append(im.convert('RGB'))
    if pdf_images:
        pdf_images[0].save(output_pdf, save_all=True, append_images=pdf_images[1:])
        if debug:
            print(f"Successfully created PDF: {output_pdf}")
    else:
        print("No images to save as PDF.")


# Main program
debug = input("Debug mode on? y(es) ")
debug =  (debug == 'y' or debug == 'yes')
video_file = input("Video to scan (default is ./data/nb.mp4): ") or "./data/nb.mp4"
if debug:
    print("Scanning", video_file, "...")
is_notebook = input("Is video a notebook? y(es) or n(o): ")
is_notebook = (is_notebook == "y" or is_notebook == "yes")
fps = int(input("Frames per second to extract (default 5): ") or 5)
if debug:
    print(f"Extracting frames at {fps} fps...")

fst = time.time()

stt = time.time()
output_dir = "./trash"
os.makedirs(output_dir, exist_ok=True)

for d in ["frames"]:
    os.makedirs(output_dir + "/" + d, exist_ok=True)

frames = video_to_frames(video_file, fps)
if debug:
    print("Extracting frames took: ", time.time()-stt, "seconds")

model = FastSAM('FastSAM-x.pt') # FastSAM.pt or FastSAM-s.pt(smaller & less accurate)

stt = time.time()
fin_pg = []
print("")
for i in range(0, len(frames)):
    masks = genfs_numpy(frames[i])
    progress_bar(i+1, len(frames), length=50)

    #pgs = find_pages_nb(masks, 0.20, 0.50, 0.20)
    #pgs = filter_covered_pages(get_pages(masks, 0.20, 0.10), 0.20)
    pgs = filter_covered_pages(find_pages_nb(masks, 0.20, 0.50, 0.10), 0.20)
    fin_pg.append(pgs)
print("")
if debug:
    print("FastSAM on all frames took: ", time.time()-stt, "seconds")

stt = time.time()

goodpg = []
for i in fin_pg:
    # Need 2 pages if you are finding a notebook and 1 page if finding individual pages
    if is_notebook:
        g = (len(i) == 2)
    else:
        g = (len(i) == 1)
    for j in i:
        g = (g and mono(j, 10, 0.50))
#        g = (g and mono(j, 2, 10))
    goodpg.append(g)

if debug:
    print(goodpg)

ff = final_frames(goodpg)
if debug:
    print("Finding valid pages took: ", time.time()-stt, "seconds")
    print("Final Frames:", ff)

output_pdf = "output.pdf"
frames_to_pdf(frames, fin_pg, ff, output_pdf)
print("Saved to", output_pdf)

if debug:
    print("Total time taken: ", time.time()-fst, "seconds")

# maybe make lower bound for page size more lenient
