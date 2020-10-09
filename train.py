
"""

Editor : Emad Aljaser
Subject : Imgae Classifier
Date : 29/9/2020


"""
from train import *
import argparse
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import json
import PIL

### Create arg parse inpt...

def input_args():
    parser = argparse.ArgumentParser(description = "Image Classifier")

    parser.add_argument('--checkpoint', type= str, default = 'checkpoint.pth',
    help= 'Path to load the model')

    parser.add_argument('--image', type= str,
    default = 'flowers/test/27/image_06887.jpg', help= 'Choose image directory')

    parser.add_argument('--topk', type= int, default = 5,
    help= 'Determine top-k matches')

    parser.add_argument('--cat_file', type= str, default = 'cat_to_name.json',
    help= 'mapping file from category label to category name.')

    parser.add_argument('--gpu', action='store_true',
                    default=True,
                    dest='gpu',
                    help='Set a switch to true')

    input = parser.parse_args()

    return input

### Fuction for load the model and classifier...

def Load_model(file_path, device):

    checkpoint = torch.load(file_path, map_location=lambda storage, loc: storage)

    arch = checkpoint['architechture']

    model = models.__dict__[arch](pretrained=True)

    hidden_units_1 = checkpoint['hidden_units_1']
    hidden_units_2 = checkpoint['hidden_units_2']
    learning_rate = checkpoint['learning_rate']

    model, classifier, criterion, optimizer = initiate_DNN(model, arch,
    hidden_units_1, hidden_units_2, learning_rate, device)

    model.load_state_dict(checkpoint['state_dict'])

    model.class_to_idx = checkpoint['class_to_idx']

    return model


### image preparation..

def process_image(image):

    img = PIL.Image.open(image)

    image_preprocessing = transforms.Compose([transforms.RandomRotation(35),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

    img = image_preprocessing(img).float()

    return img

### predictions

def predict(image, model, cat_file, topk, device):

    with open(cat_file, 'r') as f:
        cat_to_name = json.load(f)

    model.eval();
    model.to(device);

    img = process_image(image)
    img = img.unsqueeze (dim = 0)

    img = img.to(device)

    with torch.no_grad ():

        logps = model.forward(img)

        outputs = torch.exp(logps)
        top_probs, top_indices = outputs.topk(topk)

        top_probs = top_probs.cpu().numpy().tolist()[0]
        top_indices = top_indices.cpu().numpy().tolist()[0]

        idx_to_class = {val: key for key, val in
                        model.class_to_idx.items()}

        top_classes = [idx_to_class[label] for label in top_indices]
        top_flowers = [cat_to_name[label] for label in top_classes]

    return top_probs, top_classes, top_flowers

### print function...

def print_result(top_probs, top_flowers):
    print("\n\nTop 5 classes are:\n")
    i = 0
    for x, y in zip(top_probs, top_flowers):
        i +=1
        print("{}. Flower: {:25} >>> Probability: {:.2f}%".format(i, y, x*100))


### main Fuction...

def main():

    input = input_args()

    device = check_gpu(gpu=True)


    model = Load_model(input.checkpoint, device)


    top_probs, top_classes, top_flowers = predict(input.image, model,
     input.cat_file, input.topk, device)

    print_result(top_probs, top_flowers)



if __name__ == "__main__":
    main()
