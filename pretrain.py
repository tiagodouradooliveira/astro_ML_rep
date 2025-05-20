import torch
from torch import nn
from tqdm import tqdm
import pandas
import os
from utils import load_pretrain_data, model_picker

class CustomMAE(nn.Module):
    def __init__(self):
        super(CustomMAE, self).__init__()
        self.mae = nn.L1Loss()

    def forward(self, inputs, targets):
        missing_value = 99
        mask = torch.tensor(targets != missing_value, dtype=torch.float32)
        return self.mae(inputs*mask, targets*mask)

def pretrain(model, survey_table, glorot_init, training_path, 
             testing_path, batch_size, transform, use_imgnet_weights,
             saved_models_path):
    if use_imgnet_weights:
        model = model_picker(model, True, 3, glorot_init)
    else:
        model = model_picker(model, False, 3, glorot_init)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    print("Getting pretrain routine dataloaders...")
    train_dataloader, val_dataloader = load_pretrain_data(
        survey_table, training_path, testing_path, batch_size, transform
    )
    print("Success.")

    # Implement later
    if model == "inceptionresnetv2" or model == "inception_v3" or model == "resnext50":
        torch.save(obj=model.state_dict(), f=os.join(saved_models_path, "checkpoint_pretrain.pth"))
        return 

    criterion = CustomMAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    best_loss = 1e6

    model.features.requires_grad_(False)

    for epoch in tqdm(range(100)):
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
            torch.save(obj=model.state_dict(), f=os.join(saved_models_path, "checkpoint_pretrain.pth"))