---
layout: page
title: Visualize Random COCO
description: Visualize N random Samples 
img: assets/img/jn.png
importance: 4
category: tools
---

This is a small script to generate samples with bounding boxes and corresponding labels.

# COCO Format 
```python
from pycocotools.coco import COCO
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


def find_name_by_id(lst, search_id):
    for item in lst:
        if item.get("id") == search_id:
            return item.get("name")
    return None


def save_annotated_images(coco, image_ids, num_images, visdrone_folder, output_folder):
    for i in range(num_images):
        image_id = random.choice(image_ids)
        image_data = coco.loadImgs(image_id)[0]
        image_path = os.path.join(visdrone_folder, image_data['file_name'])
        annotations = coco.loadAnns(coco.getAnnIds(image_id))
        
        # Load the image
        image = plt.imread(image_path)
        
        # Plot annotations
        fig, ax = plt.subplots()
        ax.imshow(image)
        for ann in annotations:
            bbox = ann['bbox']
            rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                     linewidth=1, edgecolor='r', facecolor='none')
            plt.gca().add_patch(rect)
            
            category_id = ann['category_id']
            category_name =  find_name_by_id(coco.dataset["categories"], category_id)# [category_id - 1]
            plt.text(bbox[0], bbox[1] - 2, category_name, fontsize=8,
                     color='r', verticalalignment='top', bbox={'color': 'white', 'alpha': 0.7, 'pad': 2})
        
        
        ax.axis('off')
        output_path = os.path.join(output_folder, f'annotated_{i+1}.jpg')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


# Specify the COCO annotations file path and output folder
coco_annotations_path = '../scaled_visdrone_mot_train.json'
visdrone_folder = '../scaled_label_train'
output_folder = "."

# Initialize COCO API
coco = COCO(coco_annotations_path)

# Get all image IDs
image_ids = coco.getImgIds()

# Save annotated images for 5 random images
save_annotated_images(coco, image_ids, num_images=5, visdrone_folder=visdrone_folder, output_folder=output_folder)
    
```