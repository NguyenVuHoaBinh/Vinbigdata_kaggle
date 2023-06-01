# import library
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from ensemble_boxes import *

import os
import csv
import pandas as pd
import numpy as np
import shutil
import random

def random_increase(value):
    max_increase = min(0.9 - value, value)
    increase = random.uniform(0, max_increase)
    new_value = value + increase
    return new_value

dict = {
    0: "Aotic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding"
}

dict_vie = {
    0: "Phình động mạch chủ",
    1: "Xẹp phổi",
    2: "Vôi hoá/ Lắng cặn canxi",
    3: "Tim to",
    4: "Đông đặc phổi",
    5: "Phổi kẽ",
    6: "Thâm nhiễm phổi",
    7: "Phế nang viêm",
    8: "U phổi",
    9: "Mô bất thường khác",
    10: "Tràn dịch màng phổi",
    11: "Dày dịch màng phổi",
    12: "Xẹp phổi",
    13: "Xơ phổi",
    14: "No finding"}
def YOLO_inf(image_path):
    # Create the model object
    model_0 = YOLO(r"E:\XR-AI\model\fold1\best.pt")
    model_1 = YOLO(r"E:\XR-AI\model\fold2\best.pt")
    model_2 = YOLO(r"E:\XR-AI\model\fold3\best.pt")
    model_3 = YOLO(r"E:\XR-AI\model\fold4\best.pt")
    model_4 = YOLO(r"E:\XR-AI\model\fold5\best.pt")
    model_5 = YOLO(r"E:\XR-AI\model\fold6\best.pt")
    model_6 = YOLO(r"E:\XR-AI\model\fold8\best.pt")
    model_list = [model_0, model_1, model_2, model_3, model_4, model_5, model_6]

    # Start inference
    for i in model_list:
        result = i.predict(source=image_path, iou=0.4, conf=0.4, save_txt=True, save_conf=True, augment=True,
                           project="temp", name="model")

    # Colect all the result
    directory = r'E:\XR-AI\temp'
    output_file = r'E:\XR-AI\temp\file.csv'
    rows = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    contents = f.read()
                for line in contents.split('\n'):
                    data = line.split()
                    if len(data) == 6:
                        rows.append(data)
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    return output_file, image_path


def bboxes(csv_file, image_width, image_height):
    # Load the CSV file into a pandas dataframe
    df = pd.read_csv(csv_file, header=None,
                     names=['class', 'x_center', 'y_center', 'width', 'height', 'confidence'])

    # Convert the dataframe to a format that is compatible with the ensemble_boxes library
    boxes_list = []
    scores_list = []
    labels_list = []

    unique_classes = df['class'].unique()

    for cls in unique_classes:
        class_df = df[df['class'] == cls]

        # Transform the bounding boxes from x_center, y_center, width, height to xmin, ymin, xmax, ymax format
        width = class_df['width'].values * image_width
        height = class_df['height'].values * image_height
        x_center = class_df['x_center'].values * image_width
        y_center = class_df['y_center'].values * image_height
        x_min = x_center - (width / 2)
        y_min = y_center - (height / 2)
        x_max = x_center + (width / 2)
        y_max = y_center + (height / 2)

        # Normalize the bounding box coordinates
        x_min_norm = x_min / image_width
        y_min_norm = y_min / image_height
        x_max_norm = x_max / image_width
        y_max_norm = y_max / image_height

        boxes = np.column_stack([x_min_norm, y_min_norm, x_max_norm, y_max_norm])
        scores = class_df['confidence'].values.tolist()
        labels = [cls] * len(boxes)

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Perform non-maximum suppression
    iou_thr = 0.6
    skip_box_thr = 0.0001
    boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=iou_thr,skip_box_thr=skip_box_thr)
    return boxes, scores, labels


def draw_bbox(image_path, boxes, scores, labels):
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Convert the boxes from normalized xmin, ymin, xmax, ymax to pixel values
    width, height = image.size
    boxes = np.array(boxes)
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height

    # Create a PIL draw object
    draw = ImageDraw.Draw(image)

    # Plot the bounding boxes on the image
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = np.round(box).astype(int)

        # Draw the box on the image
        draw.rectangle([(x1, y1), (x2, y2)], outline=(255, 0, 0), width=4)

        # Add the label and score next to the box
        label_text = dict_vie[int(labels[i])] + f", Score: {scores[i]:.2f}"
        # Set the font parameters
        font_path = r"E:\XR-AI\2BYTE\font-times-new-roman.ttf"
        font_size = 24
        font_color = (0, 0, 255)  # Blue color
        background_color = (255, 255, 255)  # White color

        # Load the font
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the size of the text
        text_width, text_height = draw.textsize(label_text, font=font)

        # Calculate the positions for text and background
        text_position = (x1, y2 )
        bg_position = (x1, y2 )
        bg_size = (text_width, text_height)

        # Draw the background rectangle
        draw.rectangle([bg_position, (bg_position[0] + bg_size[0], bg_position[1] + bg_size[1])], fill=background_color)

        # Draw the text on the image
        draw.text(text_position, label_text, fill=font_color, font=font)

    # Save the image with the bounding boxes
    output_path = r"E:\XR-AI\temp\annotated_image.png"
    image.save(output_path)
    return output_path


def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' deleted successfully.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}.")


if __name__ == '__main__':
    # delete_folder(r"E:\XR-AI\temp")
    #result_csv, image_path = YOLO_inf(r"E:\XR-AI\datasets\vinbigdata\train\b76cca59757045fa3cedad238c878354.png")
    # boxes, scores, labels = bboxes(result_csv, 1024, 1024)
    #output = draw_bbox(image_path, boxes, scores,labels)
    model_best = YOLO(r"E:\XR-AI\model\fold2\best.pt")
    x=model_best.predict(source=r"E:\XR-AI\datasets\vinbigdata\train\b76cca59757045fa3cedad238c878354.png", iou=0.4, conf=0.4, save_txt=True, save_conf=True, augment=True,save=True, imgsz= 640,
           project="temp", name="model")

