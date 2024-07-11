from tqdm import tqdm

import cv2
import numpy as np

import json
import os
import shutil


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

    def append_txt(self, label_dir="datasets/selectstar1", image_size=None):
        image_id = len(self.image_list)
        label_id = len(self.label_list)
        label_files = [f for f in np.sort(os.listdir(label_dir)) if f.endswith(".json")]

        for label_file in tqdm(label_files):
            skip_image = False

            image_file = label_file.replace(".json", ".jpg")
            if image_size is None:
                image = cv2.imread(f"{self.image_dir}/{image_file}")
                image_size = image.shape[:2]
            image_txt = {"id": image_id, "file_name": image_file, "height": image_size[0], "width": image_size[1]}

            with open(f"{label_dir}/{label_file}") as f:
                labels = json.load(f)

            for object in labels["shapes"]:
                label_txt = dict()
                label_txt["image_id"] = image_id
                label_txt["label_id"] = label_id
                label_id += 1

                if object["label"].lower() == "ball":
                    label_txt["category_id"] = self.ball_id
                else:
                    label_txt["category_id"] = self.person_id

                if object["label"].lower() not in self.unique_classes:
                    self.unique_classes.append(object["label"].lower())

                try:
                    (x1, y1), (x2, y2) = object["points"]
                    label_txt["bbox"] = [x1, y1, x2 - x1, y2 - y1]
                    # label_txt["area"] = (x2 - x1) * (y2 - y1)
                    self.label_list.append(label_txt)
                except ValueError:
                    skip_image = True
                    continue

            if not skip_image:
                self.image_list.append(image_txt)
                image_id += 1

    def save_txt(self, save_path="datasets/soccer/annotations/instances_train2017.json"):
        coco_txt = {"images": self.image_list, "annotations": self.label_list, "categories": self.category_list}
        with open(save_path, "w") as f:
            json.dump(coco_txt, f)
