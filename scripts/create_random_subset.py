import os

from PIL import Image
import random

num_imgs = 50

sub_dir = "imgs_sun_data_3"

uncleaned_dir = os.path.join('uncleaned_images', sub_dir) 
cleaned_dir = os.path.join('cleaned_images', sub_dir) 

if not os.path.exists(cleaned_dir):
    os.makedirs(cleaned_dir)

files = os.listdir(uncleaned_dir)

files_subset = random.sample(files, num_imgs)
for file in files_subset:
    im = Image.open(os.path.join(uncleaned_dir, file))
    im.save(os.path.join(cleaned_dir, file))

