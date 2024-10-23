import os
import xml.etree.ElementTree as ET
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
def parse_xml(xml_file, image_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    if size is not None:
        width = int(float(size.find('width').text))
        height = int(float(size.find('height').text))
    else:
        image = cv2.imread(image_file)
        height, width, _ = image.shape
    objects = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.find('xmin').text))
        ymin = int(float(bndbox.find('ymin').text))
        xmax = int(float(bndbox.find('xmax').text))
        ymax = int(float(bndbox.find('ymax').text))
        objects.append([xmin, ymin, xmax, ymax])
    return objects, width, height



def process_dataset(dataset_dir):
    image_files = []
    labels = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg'):
            image_path = os.path.join(dataset_dir, filename)
            xml_path = os.path.join(dataset_dir, filename.replace('.jpg', '.xml'))
            if os.path.exists(xml_path):
                objects, width, height = parse_xml(xml_path, image_path)
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Cannot read image: {image_path}")
                    continue
                image_files.append(image_path)
                labels.append({
                    "image": image_path,
                    "bboxes": objects,
                    "image_width": width,
                    "image_height": height
                })

    return image_files, labels
dataset_dir = 'test'  
image_files, labels = process_dataset(dataset_dir)

def preprocess_data(image_files, labels, img_size=(416, 416), max_boxes=10):
    X = []
    y = []
    for label in labels:
        image_path = label["image"]
        image = cv2.imread(image_path)
        if image is not None:
            image_resized = cv2.resize(image, img_size)
            X.append(image_resized)
            bboxes = label["bboxes"]
            if len(bboxes) > max_boxes:
                bboxes = bboxes[:max_boxes] 
            while len(bboxes) < max_boxes:  
                bboxes.append([0, 0, 0, 0])  

            y.append(np.array(bboxes))  

    return np.array(X), np.array(y)



X, y = preprocess_data(image_files, labels)



