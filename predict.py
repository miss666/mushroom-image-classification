import time

import timm
import torch
from PIL import Image
import numpy as np

import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def Predict(image_path,option):
    if option =="EfficientNet-B0":
        model = torch.load('./DF20M-EfficientNet-B0_best_accuracy.pth')
    elif option =="MobileNet-V2":
        model = torch.load('.s/DF20M-MobileNet-V2_best_accuracy.pth')
    elif option =="SE-ResNext101_32x4d":
        model = torch.load('./DF20M-SE_ResNext101_32x4d_best_accuracy.pth')


    #https://pytorch.org/docs/stable/torchvision/models.html
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(image_path)
    img = img.resize((224, 224), Image.BILINEAR)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = to_tensor(img)
    img = torch.unsqueeze(img, 0)  # 给最高位添加一个维度，也就是batchsize的大小
    img = img.to(device)
    t1 = time.time()
    pred = model(img)

    t2 = time.time()
    s = round(float(t2 - t1), 3)

    prob = torch.nn.functional.softmax(pred, dim=1)[0] * 100

    _, indices = torch.sort(pred, descending=True)
    with open('imagenet_classes.txt', encoding='UTF-8') as f:
        classes = [line.strip() for line in f.readlines()]

    return [(classes[idx], prob[idx].item()) for idx in indices[0][:5]], s



