import torch
from torchvision import transforms
import os
from astropy.visualization import MinMaxInterval
from matplotlib import pyplot as plt
from glob import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import pandas as pd
from utils import toPathFormat, getObjs, load_data

def Norm(img):
    norm_tr = MinMaxInterval()
    return norm_tr(img)

def main():
    config_file = Path("config/paths.json").open('r')
    paths = json.load(config_file)

    survey_table = pd.read_csv(paths["survey_path"])
    survey_table = survey_table[(survey_table.r_iso > 13) & (survey_table.r_iso <= 22)]

    train_objs, train_objs_class, val_objs, val_objs_class, test_objs, test_objs_class = getObjs(survey_table, paths["training_path"], paths["testing_path"])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32)),
        transforms.Lambda(Norm),
    ])

    BATCH_SIZE = 128

    train_dataloader, val_dataloader, test_dataloader = load_data(
        train_objs, train_objs_class, val_objs, val_objs_class, 
        test_objs, test_objs_class, BATCH_SIZE, transform
    )

    print(f"{len(train_dataloader)} train batches of {BATCH_SIZE}, {len(val_dataloader)} val batches of {BATCH_SIZE}, " +
        f"and {len(test_dataloader)} test batches of {BATCH_SIZE}\n")

    print("Image example:")
    _data_tensor, _label_tensor = next(iter(train_dataloader))
    plt.imshow(_data_tensor[0].squeeze().numpy(), cmap='gray')
    plt.title(f"Class: {_label_tensor.numpy()[0]}");
    print()

    config_file.close()

if __name__ == "__main__":
    main()