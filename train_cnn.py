import torch
from torch import nn
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
import argparse
from utils import load_data, model_picker, opt_picker
from pretrain import pretrain

parser = argparse.ArgumentParser(description='Train CNN models')

parser.add_argument('--model', default="vgg16", type=str, help='model')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=7e-4, type=float, help='weight decay')
parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--optimizer', default="adam", type=str, help="otimizer algorithm")
parser.add_argument('--pretrain', action='store_true', default=False, help="pretrain the model")
parser.add_argument('--use_imgnet_weights', action='store_true', default=False, help="Use IMAGENET weights")
parser.add_argument('--glorot_init', action='store_true', default=False, help="use glorot initialization")

def Norm(img):
    norm_tr = MinMaxInterval()
    return norm_tr(img)

def train(model, device, opt_name, survey_table, training_path, testing_path, 
          lr, weight_decay, num_epochs, batch_size, transform, saved_models_path):
    
    print("Getting train routine dataloaders...")
    train_dataloader, val_dataloader, _ = load_data(
        survey_table, training_path, testing_path, batch_size, transform
    )
    print("Success.")

    for layer in model.features:
        if (isinstance(layer, nn.Conv2d)):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True
    for layer in model.classifier:
        if (isinstance(layer, nn.Linear)):
            layer.weight.requires_grad = True
            layer.bias.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = opt_picker(opt_name, model.params(), lr, weight_decay)

    best_loss = 1e6

    for epoch in tqdm(range(num_epochs)):
        model.train()
        val_loss = 0.0
        batch_count = 0

        for (imgs, labels, _) in train_dataloader:
            imgs, labels = imgs.to(device=device, non_blocking=True), labels.to(device=device, non_blocking=True)
            
            optimizer.zero_grad()

            logits = model(imgs)

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            for (imgs, labels, _) in val_dataloader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device=device, non_blocking=True)
                
                logits = model(imgs)

                val_loss = criterion(logits, labels)
                val_loss += loss.item()

                batch_count += 1
        
        val_loss /= batch_count

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(obj=model.state_dict(), f=os.join(saved_models_path, "checkpoint.pth"))

def main():
    args = parser.parse_args()

    config_file = Path("config/paths.json").open('r')
    paths = json.load(config_file)

    print("Reading survey table...")
    survey_table = pd.read_csv(paths["survey_path"])
    #survey_table = survey_table[(survey_table.r_iso > 13) & (survey_table.r_iso <= 22)]
    print("Survey table read.")
    print()


    BATCH_SIZE = args.batch_size

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size=(32, 32)),
        transforms.Lambda(Norm),
    ])

    model = model_picker(args.model, False, 3, args.glorot_init)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    if args.pretrain:
        pretrain(args.model, survey_table, args.glorot_init, paths["training_path"], 
                 paths["testing_path"], args.batch_size, args.use_imgnet_weights,
                 transform, paths["saved_models_path"])
        
        model.load_state_dict(torch.load(
            os.join(paths["saved_models_path"], "checkpoint_pretrain.pth"), 
            weights_only=True))
    

    train(model, device, args.optimizer, survey_table, paths["training_path"],
            paths["testing_path"], args.lr, args.weight_decay, args.epochs,
            args.batch_size, transform, paths["saved_models_path"])

    config_file.close()

if __name__ == "__main__":
    main()