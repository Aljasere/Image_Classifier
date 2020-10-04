import argparse
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
import torch.nn.functional as F

arch = "vgg13"
hidden_units = 512
learning_rate = 0.01


exec("model = models.{}(pretrained=True)".format(arch))

def initiate_DNN(arch, model, hidden_units, learning_rate):

    for param in model.parameters():
        param.requires_grad = False

    if arch[:3] in ["vgg", "den", "ale", "squ", "mob"]:
        input_features = model.classifier[0].in_features
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


    print("Initiating Model: %s, Hidden_units: %s, Learning_rate: %s" % (arch,
     hidden_units,learning_rate))

    return model, classifier, criterion, optimizer


initiate_DNN(arch, model, hidden_units, learning_rate)

print(model)
