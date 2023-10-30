# importing the module
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2   
import sys
from segment_anything import sam_model_registry, SamPredictor
import os

def show_mask(mask, ax, obj='walls', random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif obj=='walls':
        color = np.array([30/255, 144/255, 255/255, 0.6])
    elif obj=='track':
        color = np.array([255/255, 144/255, 30/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_mask_cv2(mask, obj='walls', random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif obj=='walls':
        color = np.array([30/255, 144/255, 255/255, 0.6])
    elif obj=='track':
        color = np.array([255/255, 144/255, 30/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    cv2.imshow("image", mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

# function to display the coordinates of
# of the points clicked on the image 
 

class Annotator():
    def __init__(self):
        sam_checkpoint = "test_sam/checkpoints/sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        self.device = "cuda"
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=self.device)

        self.predictor = SamPredictor(self.sam)
        self.button = None
        self.fig = plt.figure(figsize=(10,10))
        plt.ion()
        plt.show()
        plt.axis('off')


    def predict(self, img):
        self.predictor.set_image(img)
        plt.imshow(img)
        plt.pause(.001) 
        self.input_point = []
        self.input_label = []

        # cv2.imshow('image', img)

        self.button = None
        cid = self.fig.canvas.mpl_connect('key_press_event', self.button_event)
        cid2 = self.fig.canvas.mpl_connect('button_press_event', self.click_event_plt) 
        # cv2.setMouseCallback('image', self.click_event)
        mask_combined = dict()
        mask_combined['walls'] = np.zeros((1, img.shape[0], img.shape[1]), dtype=bool)
        mask_combined['track'] = np.zeros((1, img.shape[0], img.shape[1]), dtype=bool)
        while True:
            # k = cv2.waitKey(0)
            while self.button == None: 
                plt.waitforbuttonpress()
            if self.button == 'c':    
                break

            elif self.button == 'r':
                mask_combined['walls'] = np.zeros((1, img.shape[0], img.shape[1]), dtype=bool)
                mask_combined['track'] = np.zeros((1, img.shape[0], img.shape[1]), dtype=bool)
                plt.clf()
                plt.imshow(img)
            elif self.button == 'w':
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.input_point),
                    point_labels=np.array(self.input_label),
                    multimask_output=False,
                )
                mask_combined['walls'] = np.logical_or(mask_combined['walls'], masks)
                plt.clf()
                plt.imshow(img)
                show_mask(mask_combined['walls'], plt.gca(), 'walls')
                show_mask(mask_combined['track'], plt.gca(), 'track')

            elif self.button == 't':
                masks, scores, logits = self.predictor.predict(
                    point_coords=np.array(self.input_point),
                    point_labels=np.array(self.input_label),
                    multimask_output=False,
                )
                mask_combined['track'] = np.logical_or(mask_combined['track'], masks)
                plt.clf()
                plt.imshow(img)
                show_mask(mask_combined['walls'], plt.gca(), 'walls')
                show_mask(mask_combined['track'], plt.gca(), 'track')


            self.button = None
            self.input_point = []
            self.input_label = []
        #show_points(input_point, input_label, plt.gca())
        self.fig.canvas.mpl_disconnect(cid)
        self.fig.canvas.mpl_disconnect(cid2)
        plt.clf()
        mask_out = np.zeros((1, img.shape[0], img.shape[1]), dtype=np.uint8)
        mask_out[mask_combined['walls'] == True] = 1
        mask_out[mask_combined['track'] == True] = 2
        return mask_out


    def click_event_plt(self, event):  
        # checking for left mouse clicks
        if event.button == 1:
            print(event.xdata, ' ', event.ydata)
            self.input_point.append([event.xdata, event.ydata])
            self.input_label.append(1)

    def click_event(self, event, x, y, flags, params):  
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            print(x, ' ', y)
            self.input_point.append([x, y])
            self.input_label.append(1)

    def button_event(self, event):
        self.button = event.key


def main():
    img_dir = 'cleaned_images/imgs_sun_data_2'
    out_dir = 'cleaned_labels/imgs_sun_data_2'
    image_files = os.listdir(img_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    annotated_files = os.listdir(out_dir)
    annotate = Annotator()

    for file in image_files:
        if file not in annotated_files:
            print(file)
            img = cv2.imread(os.path.join(img_dir, file), 1)
            out = annotate.predict(img)
            # out = cv2.resize(np.array(out[0], dtype=np.uint8), (128, 128))
            cv2.imwrite(os.path.join(out_dir, file), out[0])
# driver function
if __name__=="__main__":
    main()

