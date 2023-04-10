import time

import timm
import torch
from PIL import Image
import numpy as np

import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def getModel(architecture_name, target_size, pretrained=False):
    net = timm.create_model(architecture_name, pretrained=pretrained)
    net_cfg = net.default_cfg
    last_layer = net_cfg['classifier']
    num_ftrs = getattr(net, last_layer).in_features
    setattr(net, last_layer, nn.Linear(num_ftrs, target_size))
    return net
def dist():
    # 从csv表读入图像路径
    train_metadata = pd.read_csv('E:\data1\DF20M-train_metadata_PROD_D.csv', encoding='ISO-8859-1')
    train_metadata.Habitat = train_metadata.Habitat.replace(np.nan, 'unknown', regex=True)
    """Extracting Species distribution"""
    class_priors = np.zeros(len(train_metadata['class_id'].unique()))
    for species in train_metadata['class_id'].unique():
        class_priors[species] = len(train_metadata[train_metadata['class_id'] == species])#同一个class_id图片的数量
    class_priors = class_priors/sum(class_priors)#每个class_id在总图片中的占比

    #Extracting species-habitat distribution
    habitat_distributions = {}
    for _, observation in train_metadata.iterrows():
        habitat = observation.Habitat
        class_id = observation.class_id
        if habitat not in habitat_distributions:
            habitat_distributions[habitat] = np.zeros(len(train_metadata['class_id'].unique()))
        else:
            habitat_distributions[habitat][class_id] += 1

    for key, value in habitat_distributions.items():
        if sum(value) == 0:
            habitat_distributions[key] = value * 0
        else:
            habitat_distributions[key] = value / sum(value)

        # 每个栖息地对应的每个物种的数量占该栖息地总物种数量的比例

    """Extracting species-month distribution"""
    month_distributions = {}
    for _, observation in train_metadata.iterrows():#遍历表格每一行（序号，值）
        month = str(observation.month)
        class_id = observation.class_id
        if month not in month_distributions:
            month_distributions[month] = np.zeros(len(train_metadata['class_id'].unique()))
        else:
            month_distributions[month][class_id] += 1 #某个月对应的某个物种数量+1
    for key, value in month_distributions.items():
        month_distributions[key] = value / sum(value)#每个月对应的每个物种的数量占该月总物种数量的比例

    #Extracting species-substrate distribution
    substrate_distributions = {}
    for _, observation in train_metadata.iterrows():
        substrate = observation.Substrate
        class_id = observation.class_id
        if substrate not in substrate_distributions:
            substrate_distributions[substrate] = np.zeros(len(train_metadata['class_id'].unique()))
            substrate_distributions[substrate][class_id] = 1
        else:
            substrate_distributions[substrate][class_id] += 1

    for key, value in substrate_distributions.items():

        if sum(value) == 0:
            substrate_distributions[key] = value * 0
        else:
            substrate_distributions[key] = value / sum(value)
        # 每个基底对应的每个物种的数量占该基底总物种数量的比例
    return class_priors,habitat_distributions,month_distributions,substrate_distributions

def Predict(image_path,option):
    if option =="EfficientNet-B0":
        model = torch.load('../models/DF20M-EfficientNet-B0_best_accuracy.pth')
    elif option =="MobileNet-V2":
        model = torch.load('../models/DF20M-MobileNet-V2_best_accuracy.pth')
    elif option =="SE-ResNext101_32x4d":
        model = torch.load('../models/DF20M-SE_ResNext101_32x4d_best_accuracy.pth')


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



