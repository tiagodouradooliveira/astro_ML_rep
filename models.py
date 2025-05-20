import torch
from torch import nn
from torchvision import models

def init_clf_params(model, glorot, name_clf):
    gain = nn.init.calculate_gain('relu')

    if name_clf == "classifier":
        for layer in model.classifier:
                if isinstance(layer, nn.Linear):
                    if glorot:
                        nn.init.xavier_normal_(layer.weight, gain)
                    nn.init.constant_(layer.bias, 0.01)
    elif name_clf == "fc":
        for layer in model.fc:
                if isinstance(layer, nn.Linear):
                    if glorot:
                        nn.init.xavier_normal_(layer.weight, gain)
                    nn.init.constant_(layer.bias, 0.01)
    else:
        for layer in model.last_linear:
                if isinstance(layer, nn.Linear):
                    if glorot:
                        nn.init.xavier_normal_(layer.weight, gain)
                    nn.init.constant_(layer.bias, 0.01)

    return model

def alexnet(default_weights, num_classes, glorot):
    model = models.alexnet(weights='DEFAULT') if default_weights else models.alexnet()

    model.features[0].in_channels = 12

    model.classifier = nn.Sequential(
        nn.Linear(256 * 6 * 6, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    model = init_clf_params(model, glorot, "classifier")

    return model

def vgg16(default_weights, num_classes, glorot):
    model = models.vgg16(weights='DEFAULT') if default_weights else models.vgg16()

    model.features[0].in_channels = 12

    model.classifier[-1] = nn.Linear(4096, num_classes)

    model = init_clf_params(model, glorot, "classifier")

    return model

def inception_v3(default_weights, num_classes, glorot):
    model = models.inception_v3(weights='DEFAULT') if default_weights else models.inception_v3()

    model.Conv2d_1a_3x3.conv.in_channels = 12

    model.fc = nn.Sequential(
        nn.Linear(2048, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    model = init_clf_params(model, glorot, "fc")

    return model

def resnext50(default_weights, num_classes, glorot):
    model = models.resnext50_32x4d(weights='DEFAULT') if default_weights else models.resnext50_32x4d()

    model.conv1.in_channels = 12

    model.fc = nn.Sequential(
        nn.Linear(512 * 4, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    model = init_clf_params(model, glorot, "fc")

    return model

def densenet121(default_weights, num_classes, glorot):
    model = models.densenet121(weights='DEFAULT') if default_weights else models.densenet121()

    model.features.conv0.in_channels = 12

    model.classifier = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )

    model = init_clf_params(model, glorot, "classifier")

    return model