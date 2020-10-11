
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
    """
    This function use python's argparse module and takes 5 comanand lines can be used by the user in the terminal window, all arguments have default input can be used to run the program.
    Command Line Arguments:
    1. Save directory as checkpoint with default value 'checkpoint.pth'.
    2. Image directory for inference as --image with default value.
    3. Catogery file as --cat_file with default value 'cat_to_name.json'.
    4. Save directory as --save_dir with default value 'checkpoint.pth'.
    5. GPU boolean selection as --gpu with default value 'True'
    This function return input as parser.parse_args(), function stores and excutes all command line arguments.
    """
    parser = argparse.ArgumentParser(description = "Image Classifier")

    parser.add_argument('--checkpoint',
                        type= str,
                        default = 'checkpoint.pth',
                        help= 'Path to load the model'
                        )

    parser.add_argument('--image',
                        type= str,
                        default = 'flowers/test/27/image_06887.jpg',
                        help= 'Choose image directory'
                        )

    parser.add_argument('--topk',
                        type= int,
                        default = 5,
                        help= 'Determine top-k matches'
                        )

    parser.add_argument('--cat_file',
                        type= str,
                        default = 'cat_to_name.json',
                        help= 'mapping file from category label to category name.'
                        )

    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        dest='gpu',
                        help='Set a switch to true to use the GPU'
                        )


    input = parser.parse_args()

    return input

### Fuction for load the model and classifier...

def Load_model(file_path, device):
    """
    Fuction takes two arguments to load the model and initiates new classifier the has been used for training.
    Command Line Arguments:
    1. File path as file_path
    2. Device
    Fuction returns model
    """

    checkpoint = torch.load(file_path, map_location=lambda storage,
                loc:storage) # To load the model from gpu and vice versa

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
    """
    Function takes one argument and use transforms module for image process.
    Command Line Arguments:
    1. Image path as image.
    returns Tensoe Image
    """

    img = PIL.Image.open(image)

    image_preprocessing = transforms.Compose([transforms.RandomRotation(35),
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
                        ])

    img = image_preprocessing(img).float()

    return img

### predictions

def predict(image, model, cat_file, topk, device):
    """
    Function takes 5 arguments for inference and use device selection for inference.
    Command Line Arguments:
    1. Image path as image.
    2. Model.
    3. Catogery file as cat_file.
    4. Top k as topk.
    5. Device.
    This Fuction returns top_probs, top_classes, top_flowers
    """

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
    """
    Fuction print out top 5 classes that has been predicted.
    Command Line Arguments:
    1. Top 5 Probabilities as top_probs.
    2. Top 5 flowers as top_flowers.
    This function returns None.
    """
    print("\n\nTop 5 classes are:\n")
    i = 0
    for x, y in zip(top_probs, top_flowers):
        i +=1
        print("{}. Flower: {:25} >>> Probability: {:.2f}%".format(i, y, x*100))


### main Fuction...

def main():

    input = input_args()

    device = check_gpu(input.gpu)

    model = Load_model(input.checkpoint, device)


    top_probs, top_classes, top_flowers = predict(input.image, model,
                input.cat_file, input.topk, device)

    print_result(top_probs, top_flowers)



if __name__ == "__main__":
    main()
