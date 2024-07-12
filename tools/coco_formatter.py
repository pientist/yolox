import json
import os
import sys

if not os.getcwd() in sys.path:
    sys.path.append(os.getcwd())

from tqdm import tqdm

import cv2
import numpy as np


class COCOFormatter:
    def __init__(self, image_dir="datasets/soccer/train2017"):
        self.image_dir = image_dir

        self.image_list = []
        self.label_list = []
        self.unique_classes = []

        with open("datasets/categories.json") as f:
            self.category_list = json.load(f)["categories"]

        for c in self.category_list:
            if c["name"] == "person":
                self.person_id = c["id"]
            elif c["name"] == "sports ball":
                self.ball_id = c["id"]

    def append_txt(self, label_dir="datasets/selectstar1", image_size=None, resize_images=False):
        label_files = [f for f in np.sort(os.listdir(label_dir)) if f.endswith(".json")]

        for label_file in tqdm(label_files):
            skip_image = False
            crop_x, crop_y = 0, 0

            image_file = label_file.replace(".json", ".jpg")
            if image_size is None or resize_images:
                image = cv2.imread(f"{self.image_dir}/{image_file}")
                if resize_images and (image.shape[0] > image_size[0] or image.shape[1] > image_size[1]):
                    crop_y = max(image.shape[0] - image_size[0], 0)
                    crop_x = max(image.shape[1] - image_size[1], 0)
                    image = image[crop_y:, crop_x:]
                    cv2.imwrite(f"{self.image_dir}/{image_file}", image)
                image_size = image.shape[:2]

            image_id = len(self.image_list)
            image_txt = {"id": image_id, "file_name": image_file, "height": image_size[0], "width": image_size[1]}

            with open(f"{label_dir}/{label_file}") as f:
                labels = json.load(f)

            for object in labels["shapes"]:
                label_txt = dict()
                label_txt["id"] = len(self.label_list)
                label_txt["image_id"] = image_id
                label_txt["iscrowd"] = 0

                if object["label"].lower() == "ball":
                    label_txt["category_id"] = self.ball_id
                else:
                    label_txt["category_id"] = self.person_id

                if object["label"].lower() not in self.unique_classes:
                    self.unique_classes.append(object["label"].lower())

                try:
                    (x1, y1), (x2, y2) = object["points"]
                    if x1 >= crop_x and y1 >= crop_y:
                        label_txt["bbox"] = [x1 - crop_x, y1 - crop_y, x2 - x1, y2 - y1]
                        label_txt["area"] = (x2 - x1) * (y2 - y1)
                        self.label_list.append(label_txt)
                except ValueError:
                    skip_image = True
                    continue

            if not skip_image:
                self.image_list.append(image_txt)

    def translate_bbox_labels(self, x=0, y=0):
        new_label_list = []
        for label in self.label_list:
            label["bbox"][0] += x
            label["bbox"][1] += y
            new_label_list.append(label)
        self.label_list = new_label_list

    def save_txt(self, save_path="datasets/soccer/annotations/instances_train2017.json"):
        coco_txt = {"images": self.image_list, "annotations": self.label_list, "categories": self.category_list}
        with open(save_path, "w") as f:
            json.dump(coco_txt, f)
