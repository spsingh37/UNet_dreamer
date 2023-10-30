import os
from PIL import Image
import random
random.seed(0)
img_dir = 'cleaned_images'
label_dir = 'cleaned_labels'

img_sub_dirs = os.listdir(img_dir)
label_sub_dirs = os.listdir(label_dir)

dataset_root = 'dataset_large'
if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)
train_count = 0
val_count = 0
for dir in img_sub_dirs:
    if dir in label_sub_dirs:
        imgs = os.listdir(os.path.join(img_dir, dir))
        labels = os.listdir(os.path.join(label_dir, dir))
        for img in imgs:
            if img in labels:
                im = Image.open(os.path.join(img_dir, dir, img))
                label = Image.open(os.path.join(label_dir, dir, img))
                
                if random.random() < 0.8:
                    im.save(os.path.join(dataset_root, 'train', 'images', 'frame'+str(train_count).zfill(4)+'.png'))
                    label.save(os.path.join(dataset_root, 'train', 'targets', 'frame'+str(train_count).zfill(4)+'.png'))
                    train_count += 1
                else:
                    im.save(os.path.join(dataset_root, 'val', 'images', 'frame'+str(val_count).zfill(4)+'.png'))
                    label.save(os.path.join(dataset_root, 'val', 'targets', 'frame'+str(val_count).zfill(4)+'.png'))
                    val_count += 1

