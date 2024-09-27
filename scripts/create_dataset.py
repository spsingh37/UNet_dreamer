import os
from PIL import Image
import random
random.seed(0)
img_dir = 'seg-cam_images'

img_sub_dirs = os.listdir(img_dir)

dataset_root = 'dataset_large'
if not os.path.exists(dataset_root):
    os.makedirs(dataset_root)
train_count = 0
val_count = 0
for dir in img_sub_dirs:
    imgs = os.listdir(os.path.join(img_dir, dir))
    i=0
    for_train = True
    for img in imgs:
        
        if i%2==0:
            # print("image: " + img)
            if i%100==0:
                print("train_count: ", train_count)
            im = Image.open(os.path.join(img_dir, dir, img))
            i+=1
            
            if random.random() < 0.95:
                im.save(os.path.join(dataset_root, 'train', 'images', 'frame'+str(train_count).zfill(4)+'.png'))
                for_train = True
            else:
                im.save(os.path.join(dataset_root, 'val', 'images', 'frame'+str(val_count).zfill(4)+'.png'))
                for_train = False
        else:
            # print("label: " + img)
            # print("val_count: ", val_count)
            label = Image.open(os.path.join(img_dir, dir, img))
            i+=1
            if for_train:
                label.save(os.path.join(dataset_root, 'train', 'targets', 'frame'+str(train_count).zfill(4)+'.png'))
                train_count += 1
            else:
                label.save(os.path.join(dataset_root, 'val', 'targets', 'frame'+str(val_count).zfill(4)+'.png'))
                val_count += 1

