import torch
from torch import nn, optim
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
    regex_str = "(^.*?/g/)|(^.*?/i/)|(^.*?/J0378/)|(^.*?/J0395/)|(^.*?/J0410/)|(^.*?/J0430/)|(^.*?/J0515/)|(^.*?/J0660/)|(^.*?/J0861/)|(^.*?/r/)|(^.*?/u/)|(^.*?/z/)"
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
    
def toPathFormat(obj, obj_class, band, prefix_path):
    return Path(os.path.join(prefix_path, class_decode[obj_class], band, (obj.replace('\'', '_') + ".npy")))

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

class PretrainDataset(data.Dataset):
    def __init__(self, objs, survey_table, path, transforms):
        self.ids = objs
        self.survey_table = survey_table
        self.path = path
        self.transforms = transforms

    def __getitem__(self, index):
        _img = np.array([])
        _id = self.ids[index]
        _label = self.survey_table.loc[_id].r_iso
        _class = self.survey_table.loc[_id].target
        for band in BANDS:
            path = toPathFormat(_id, _class, band, self.path)
            _img.append(np.load(path).astype(np.float32))

        if self.transforms is not None:
            return self.transforms(_img), _label, _id

        else:
            return _img, _label, _id

    def __len__(self):
        return len(self.ids)

class AstroDataset(data.Dataset):
    def __init__(self, objs, objs_class, path, transforms):
        self.ids = objs
        self.objs_class = objs_class
        self.path = path
        self.transforms = transforms

    def __getitem__(self, index):
        _img = np.array([])
        _label = self.objs_class[index]
        _id = self.ids[index]
        for band in BANDS:
            path = toPathFormat(_id, _label, band, self.path)
            _img.append(np.load(path).astype(np.float32))

        if self.transforms is not None:
            return self.transforms(_img), _label, _id

        else:
            return _img, _label, _id

    def __len__(self):
        return len(self.ids)
    
def load_pretrain_data(survey_table, training_path, testing_path, batch_size, transform):
    train_objs, train_objs_class, val_objs, val_objs_class, test_objs, test_objs_class = getObjs(survey_table, training_path, testing_path)

    if (isinstance(survey_table.index[0], int)):
        survey_table = survey_table.set_index('ID')

    train_dataset = PretrainDataset(train_objs, survey_table, training_path, transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = PretrainDataset(val_objs, survey_table, training_path, transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader

def load_data(survey_table, training_path, testing_path, batch_size, transform):
    train_objs, train_objs_class, val_objs, val_objs_class, test_objs, test_objs_class = getObjs(survey_table, training_path, testing_path)

    train_dataset = AstroDataset(train_objs, train_objs_class, training_path, transform)
    train_dataloader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=8, pin_memory=True)

    val_dataset = AstroDataset(val_objs, val_objs_class, training_path, transform)
    val_dataloader = data.DataLoader(val_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    test_dataset = AstroDataset(test_objs, test_objs_class, testing_path, transform)
    test_dataloader = data.DataLoader(test_dataset, batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return train_dataloader, val_dataloader, test_dataloader

def model_picker(model_name, default_weights=False, num_classes=3, glorot=True):
    if model_name == "alexnet":
        model = models.alexnet(default_weights, num_classes, glorot)

    elif model_name == "vgg16":
        model = models.vgg16(default_weights, num_classes, glorot)
        
    elif model_name == "inception_v3":
        model = models.inception_v3(default_weights, num_classes, glorot)

    elif model_name == "resnext50":
        model = models.resnext50(default_weights, num_classes, glorot)

    elif model_name == "densenet121":
        model = models.densenet121(default_weights, num_classes, glorot)

    else:
        raise ValueError(f"Model {model_name} not supported.")

    return model

def opt_picker(opt_name, params, lr=1e-5, weight_decay=0):
    if opt_name == "rmsprop":
        opt = optim.RMSprop(params, lr, weight_decay)
    elif opt_name == "adam":
        opt = optim.Adam(params, lr, weight_decay)
    elif opt_name == "radam":
        opt = optim.RAdam(params, lr, weight_decay)
    else:
        raise ValueError(f"Model {opt_name} not supported.")
    
    return opt