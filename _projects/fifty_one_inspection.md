---
layout: page
title: FiftyOne Inspection
description: Remote SSH Dataset Inspection using FiftyOne 
img: assets/img/jn.jpg
importance: 1
category: tools
---
Inspect dataset using FiftyOne on a remote machine through SSH

# COCO Format 
```python
import fiftyone as fo

# because I do not want to keep track of DB-names
for dataset in fo.list_datasets():
    dataset = fo.load_dataset(dataset)
    dataset.delete()

name = "your_dataset_name"

"""
Train
"""
labels_path = " "
data_path = " "

# Import dataset by explicitly providing paths to the source media and labels
dataset = fo.Dataset.from_dir(
    dataset_type=fo.types.COCODetectionDataset,
    data_path=data_path,
    labels_path=labels_path,
    name=name,
)

dataset = fo.load_dataset(name)
session = fo.launch_app(dataset, remote=True) # remote
session.wait() 
    
```

# YOLOv5 Format 
```python

import fiftyone as fo

for dataset in fo.list_datasets():
    dataset = fo.load_dataset(dataset)
    dataset.delete()

# A name for the dataset
name = "my-dataset"

# The directory containing the dataset to import
dataset_dir = "../supervised_annotations_yolov5/DIR/" # Expected: DIR/images and DIR/LABELS 

# The type of the dataset being imported
dataset_type = fo.types.YOLOv5Dataset  # for example

#splits = ["train", "val"]
splits = ["train"]

dataset = fo.Dataset(name)
for split in splits:
    dataset.add_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.YOLOv5Dataset, # Note. 
        split=split,
        tags=split,
)


session = fo.launch_app(dataset, remote=True, port=5151) # remote
session.wait()    
```