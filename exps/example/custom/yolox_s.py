#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from yolox.exp import Exp as MyExp

import os


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "datasets/soccer"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_val2017.json"

        self.input_size = (2080, 3840)
        self.test_size = (2080, 3840)
        self.num_classes = 80

        self.max_epoch = 100
        self.data_num_workers = 4
        self.print_interval = 50
        self.eval_interval = 1
