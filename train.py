"""

Editor : Emad Aljaser
Subject : Training Imgae Classifier
Date : 29/9/2020


"""

import argparse
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
import torch.nn.functional as F


### Function that gets the following command line inputs

def input_args():

    parser = argparse.ArgumentParser(description = "Training Image Classifier")

    parser.add_argument('--data_dir', type= str, default = 'flowers',
    help= 'Determine the data file')

    parser.add_argument('--arch', type= str, default='vgg13',
    help='Specify model architechture')

    parser.add_argument('--learning_rate', type= int, default= 0.01,
    help='Specify training learning rate')

    parser.add_argument('--hidden_units', type= int, default= 512,
    help= 'Specify classifier hidden units')

    parser.add_argument('--epochs', type= int, default= 20,
    help= 'Specify number of epochs')

    parser.add_argument('--save_dir', type= str default="classifier.pth",
    help='Enter a directory to save the checkpoint file')

    input = parser.parse_args()

    return input


### Function for data preparation..

def data_loaders(data_dir, train_dir, valid_dir, test_dir):

    train = transforms.Compose([transforms.RandomRotation(35),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(data_dir + train_dir, transform= train)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64,
     shuffle=True)

    valid = transforms.Compose([transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
    [0.229, 0.224, 0.225])
    ])

    valid_set = datasets.ImageFolder(data_dir + valid_dir, transform= valid)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=64,
     shuffle=True)

    test_set = datasets.ImageFolder(data_dir + test_dir, transform= valid)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64,
     shuffle=True)

    return train_loader, valid_loader, test_loader


### Fuction that checks gpu..

def check_gpu():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device != "gpu":
        print("CPU is availabe...")
    else:
        print("GPU is availabe...")

    return device

### Fuction that initiates complete transfer DNN...

def initiate_DNN(arch, model, hidden_units, learning_rate, device):

    for param in model.parameters():
        param.requires_grad = False

    if arch[:3] in ["vgg", "den", "ale", "squ", "mob"]:
        input_features = model.classifier.in_features
    else:
        input_features = model.fc.in_features

    classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(input_features, hidden_units)),
    ('ReLu1', nn.ReLU()),
    ('Dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(hidden_units, 102)),
    ('output', nn.LogSoftmax(dim=1))
    ]))

    if arch[:3] in ["vgg", "den", "ale", "squ", "mob"]:
        model.classifier = classifier
    else:
        model.fc = classifier


    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    model.to(device);

    print("Initiating Model: %s, Hidden_units: %s, Learning_rate: %s" % (arch,
     hidden_units,learning_rate))

    return model, classifier, criterion, optimizer

### Validation function...

def Validation(model, valid_loader, criterion, device):

    with torch.no_grad():

        valid_loss = 0
        accuracy = 0

        model.eval()

        for images, labels in valid_loader:

            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)
            batch_loss = criterion(logps, labels)
            valid_loss += batch_loss.item()

            outputs = torch.exp(logps)
            predictions, k_predictions = outputs.topk(1, dim=1)
            equals = k_predictions == labels.view(*k_predictions.shape)
            accuracy += (torch.mean(equals.type(torch.FloatTensor)).item())*100

        return valid_loss, accuracy

### training Function...

def train(model, train_loader, valid_loader, epochs, criterion,
 optimizer, device):

    steps = 0
    running_loss = 0
    print_every = 10

    print("Starting training... Number of epochs: %s" % (epochs) )

    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))

        for images, labels in train_loader:
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:

                valid_loss, accuracy = Validation(model, valid_loader,
                 criterion, device)

                print("[Running Loss: {:.3f}] ".format(running_loss/steps),
                "[Valid Loss: {:.3f}] >>".format(valid_loss/len(valid_loader)),
                "Accuracy: {:.2f}%".format(accuracy/len(valid_loader))
                )

                steps = 0
                running_loss = 0
                model.train()

    return model

### Function calculate the accuracy


def accuracy_check(model, test_loader, device):
    acc_total = []

    with torch.no_grad():
        model.eval()

        for images, labels in test_loader:

        images, labels = images.to(device), labels.to(device)

        logps = model.forward(images)

        outputs = torch.exp(logps)
        predictions, k_predictions = outputs.topk(1, dim=1)
        equals = k_predictions == labels.view(*k_predictions.shape)
        accuracy = (torch.mean(equals.type(torch.FloatTensor)).item())*100

        accuracy_total.append(accuracy)

    print("Total Test Accuracy: {:.2f}%".format(sum(acc_tot)/len(test_loader)))



def save_checkpoint(arch, epochs, save_dir, model, train_loader):

    model.class_to_idx = train_loader.class_to_idx

    checkpoint = {
    'architechture': arch,
    'epochs': epochs,
    'state_dict': model.state_dict(),
    'class_to_idx': model.class_to_idx,
    }

    torch.save(checkpoint, save_dir)

    print("Model has saved successfully in %s" % (save_dir))





##
