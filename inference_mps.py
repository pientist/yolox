from tqdm import tqdm

import cv2
import numpy as np

import torch

from yolox.exp import get_exp
from yolox.utils import postprocess

import os
import pandas as pd
from typing import Tuple


def get_drop_frames(n_frames: int) -> Tuple[int, np.ndarray]:
    n_frames_total = (int(n_frames / 29.97) + 1) * 30
    minute_frames = np.arange(0, n_frames_total + 1, 1800)
    ten_minute_frames = np.arange(0, n_frames_total + 1, 18000)
    drop_frames = np.setdiff1d(minute_frames, ten_minute_frames)
    drop_frames = np.union1d(drop_frames, drop_frames + 1)
    return n_frames + len(drop_frames), drop_frames


if __name__ == "__main__":
    model_name = "yolox_s"
    exp_path = "exps/example/custom/yolox_s.py"
    ckpt_path = f"YOLOX_outputs/{model_name}/best_ckpt.pth"

    video_name = "GX010224_Trim_18.07"
    video_path = f"datasets/soccer/videos/{video_name}.mp4"
    bbox_path = f"YOLOX_outputs/{model_name}/bboxes/{video_name}.csv"

    max_frame = 3000
    step_size = 30
    min_conf = 0.1

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = "cpu"
    print(f"Device: {device}")

    exp = get_exp(exp_path, model_name)
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")

    model = exp.get_model()
    model.to(device)
    model.half()
    model.eval()
    model.load_state_dict(ckpt["model"])

    video = cv2.VideoCapture(video_path)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames, drop_frames = get_drop_frames(n_frames)

    bbox_cols = ["x1", "y1", "x2", "y2", "conf1", "conf2", "class"]
    bbox_list = []

    for i in tqdm(np.setdiff1d(np.arange(1, n_frames), drop_frames)[:max_frame]):
        video.grab()
        if i % step_size != 0:
            continue

        ret, image = video.retrieve()
        if image is None:
            break

        padded_image = np.pad(image, ((0, 1000), (0, 1920), (0, 0))).transpose(2, 0, 1)
        with torch.no_grad():
            outputs = model(torch.HalfTensor(padded_image).unsqueeze(0).to(device))
            outputs = postprocess(
                outputs,
                exp.num_classes,
                exp.test_conf,
                exp.nmsthre,
                class_agnostic=True,
            )

        image_bboxes = pd.DataFrame(outputs[0].cpu().numpy(), columns=bbox_cols)
        image_bboxes["frame"] = i
        bbox_list.append(image_bboxes)

    bboxes = pd.concat(bbox_list, ignore_index=True)
    bboxes[bbox_cols[:4]] = bboxes[bbox_cols[:4]].round(2)
    bboxes["confidence"] = bboxes["conf1"] * bboxes["conf2"]
    bboxes["class"] = bboxes["class"].map({0: "player", 32: "ball"})
    bboxes = bboxes.loc[bboxes["confidence"] > min_conf, ["frame"] + bbox_cols[:4] + ["confidence", "class"]]
    print(bboxes)

    os.makedirs("YOLOX_outputs/yolox_s/bboxes", exist_ok=True)
    bboxes.to_csv(bbox_path, index=False)
