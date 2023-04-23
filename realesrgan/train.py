# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import sys, os
curr_path = os.path.dirname(__file__)
sys.path.insert(0, curr_path)

import realesrgan.archs
import realesrgan.data
import realesrgan.models

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
