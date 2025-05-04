import torch
from torch import nn
from torch.utils import data
from torch.nn.utils import clip_grad_norm_
import os
import joblib
from matplotlib import pyplot as plt
import re
from glob import glob
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import models

torch.manual_seed(42)

class_decode = ["quasar", "star", "galaxy"]

BANDS = ["r",
        "J0378",
        "J0395",
        "J0410",
        "J0430",
        "g",
        "J0515",
        "u",
        "J0660",
        "i",
        "J0861",
        "z"]

def toTableFormat(objs):
    regex_str = "(^.*?g/)|(^.*?i/)|(^.*?J0378/)|(^.*?J0395/)|(^.*?J0410/)|(^.*?J0430/)|(^.*?J0515/)|(^.*?J0660/)|(^.*?J0861/)|(^.*?r/)|(^.*?u/)|(^.*?z/)"
    if isinstance(objs, str):
        # Remove the regex string prefix regex_str
        obj = re.sub(regex_str, '', objs)
        # Remove the suffix .npy
        obj = obj[:-4]

        obj_tmp = list(obj)
        obj_tmp[1] = '\''
        obj_tmp[-1] = '\''
        return ''.join(obj_tmp)
    else:
        formattedObjs = []
        for obj in objs:
            obj = re.sub(regex_str, '', obj)
            obj = obj[:-4]
            obj_tmp = list(obj)
            obj_tmp[1] = '\''
            obj_tmp[-1] = '\''
            formattedObjs.append(''.join(obj_tmp))
        return formattedObjs
    
# TODO: bands
def toPathFormat(objs, objs_class, path):
    formattedObjs = []
    for obj, obj_class in zip(objs, objs_class):
        formattedObjs.append(Path(os.path.join(path, class_decode[obj_class], (obj.replace('\'', '_') + ".npy"))))
    return formattedObjs

def getObjs(survey_table, training_path, testing_path):
    ''' For objects listed in the survey table, but that could not be retrieved from the remote database '''
    
    train_path_objs = []
    test_objs = []
    
    for _class in class_decode:
        train_path_objs.extend(list(glob(f"{_class}/r/*.npy", root_dir=training_path)))
        test_objs.extend(list(glob(f"{_class}/r/*.npy", root_dir=testing_path)))

    # Remove objects for which some band(s) is missing
    BANDS_WITHOUT_R = BANDS[1:]

    for obj in train_path_objs:
        for band in BANDS_WITHOUT_R:
            if not os.path.exists(obj.replace("/r/", f"/{band}/")):
                train_path_objs.remove(obj)
                break
    
    for obj in test_objs:
        for band in BANDS_WITHOUT_R:
            if not os.path.exists(obj.replace("/r/", f"/{band}/")):
                test_objs.remove(obj)
                break

    train_path_objs = toTableFormat(train_path_objs)
    test_objs = toTableFormat(test_objs)

    if (isinstance(survey_table.index[0], int)):
        survey_table = survey_table.set_index('ID')

    # Preparing for train/val/test split
    train_ids = survey_table.loc[train_path_objs]
    val_quantity = int((len(train_path_objs) + len(test_objs)) * 0.05)
    bound_idx = len(train_path_objs) - val_quantity

    train_ids = survey_table.loc[train_path_objs]
    train_objs = list(train_ids.iloc[:bound_idx].index)
    train_objs_class = list(train_ids.iloc[:bound_idx].target)

    val_objs = list(train_ids.iloc[bound_idx:].index)
    val_objs_class = list(train_ids.iloc[bound_idx:].target)

    test_objs_class = list(survey_table.loc[test_objs].target)

    return train_objs, train_objs_class, val_objs, val_objs_class, test_objs, test_objs_class

# TODO: bands
class AstroDataset(data.Dataset):
    def __init__(self, objs, objs_class, split, transforms):
        self.ids = objs
        self.img_files = toPathFormat(list(objs), objs_class, split)
        self.objs_class = objs_class
        self.transforms = transforms

    def __getitem__(self, index):
        _img = fits.getdata(self.img_files[index]).astype(np.float32)
        _label = self.objs_class[index]
        _id = self.ids[index]

        if self.transforms is not None:
            return self.transforms(_img), _label, _id

        else:
            return _img, _label, _id

    def __len__(self):
        return len(self.img_files)
    
def load_data(train_objs, train_objs_class, val_objs, val_objs_class,
              test_objs, test_objs_class, batch_size, transform):
    train_dataset = AstroDataset(train_objs, train_objs_class, "train", transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = AstroDataset(val_objs, val_objs_class, "val", transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    test_dataset = AstroDataset(test_objs, test_objs_class, "test", transform)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

def init_params(model, glorot, name_clf):
    gain = nn.init.calculate_gain('relu')

    if name_clf == "classifier":
        for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    if glorot:
                        nn.init.xavier_normal_(layer.weight, gain)
                    nn.init.constant_(layer.bias, 0.01)
    else:
        for layer in model.fc:
                if isinstance(layer, nn.Linear):
                    if glorot:
                        nn.init.xavier_normal_(layer.weight, gain)
                    nn.init.constant_(layer.bias, 0.01)

    return model



def model_picker(model_name, default_weights=False, num_classes=3, glorot=True):
    if model_name == "alexnet":
        model = models.alexnet(default_weights, num_classes, glorot)

    elif model_name == "vgg16":
        model = models.vgg16(default_weights, num_classes, glorot)

    elif model_name == "inceptionresnetv2":
        model = models.inceptionresnetv2(default_weights, num_classes, glorot)
        
    elif model_name == "inception_v3":
        model = models.inception_v3(default_weights, num_classes, glorot)

    elif model_name == "resnext50":
        model = models.resnext50(default_weights, num_classes, glorot)

    elif model_name == "densenet121":
        model = models.densenet121(default_weights, num_classes, glorot)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model