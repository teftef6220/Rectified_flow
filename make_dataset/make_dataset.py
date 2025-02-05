import os
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import pickle

# Select and Resize data 

MAX_SIZE = 512
img_dir = './make_dataset/2D_data'
new_dir = './make_dataset/resize_images'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)
Image_files_pathes = glob.glob(os.path.join(img_dir, '*.png'))
resolutions = {}


print(len(Image_files_pathes))

for i,Image_files_path in enumerate(tqdm(Image_files_pathes)):
    img = Image.open(Image_files_path)
    W,H = img.size

    # if W or H  > max size then resize to maxsize
    if W > MAX_SIZE or H > MAX_SIZE:
        if W > H:
            img_resize = img.resize((MAX_SIZE, int(H * MAX_SIZE / W)))
        else:
            img_resize = img.resize((int(W * MAX_SIZE / H), MAX_SIZE))
    else:
        # if W or H < max size then pass . Need not resize
        pass

    # Filename
    image_name = os.path.basename(Image_files_path)

    img_resize.save(os.path.join(new_dir, image_name))

    #make resolutions.pkl for ABR
    # {
    #     "path/to/images/image1.jpg": (1920, 1080),
    #     "path/to/images/image2.png": (1280, 720),
    #     "path/to/images/image3.jpeg": (2560, 1440),
    #     "path/to/images/image4.jpg": (640, 480),
    #     "path/to/images/image5.png": (800, 600),
    # }
    resolutions[image_name] = img_resize.size

with open('./make_dataset/resolutions.pkl', 'wb') as f:
    pickle.dump(resolutions, f)