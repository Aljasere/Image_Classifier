
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
from time import time


### Function that gets the following command line inputs

def input_args():
    """
    This function use python's argparse module and takes 9 comanand line can be used by the user in the terminal, all arguments have default input can be used to run the program.
    Command Line Arguments:
    1. Image folder as --data_dir with default value 'flowers'.
    2. Batch size as --batch_size for train loader with default value "64".
    3. model architechture as --arch, user can choose any form of family of vgg, densenet, alexnet, squeezenet,and mobilenet,
    with default value 'vgg13'.
    4. Learning rate as --learning_rate with default value '0.001'.
    5. and 6. Hidden units 1, Hidden units 2
    as --hidden_units_1, --hidden_units_2
    with default values '2024' and '512'.
    7. Number of epochs as --epochs with default value '10'.
    8. Save directory as --save_dir with default value 'checkpoint.pth'.
    9. GPU boolean selection as --gpu with default value 'True'
    This function return input as parser.parse_args(), function stores and excutes all command line arguments.
    """

    parser = argparse.ArgumentParser(description = "Training Image Classifier")

    parser.add_argument('--data_dir',
                        type= str,
                        default = 'flowers',
                        help= 'Determine the data file'
                        )

    parser.add_argument('--batch_size',
                        type= int,
                        default= 64,
                        help= 'Specify number of batch size'
                        )

    parser.add_argument('--arch',
                        type= str,
                        default='vgg13',
                        help='Specify model architechture'
                        )

    parser.add_argument('--learning_rate',
                        type= int,
                        default= 0.001,
                        help='Specify training learning rate'
                        )

    parser.add_argument('--hidden_units_1',
                        type= int,
                        default= 2024,
                        help= 'Specify classifier hidden units 1'
                        )

    parser.add_argument('--hidden_units_2',
                        type= int,
                        default= 512,
                        help= 'Specify classifier hidden units 2'
                        )

    parser.add_argument('--epochs',
                        type= int,
                        default= 10,
                        help= 'Specify number of epochs'
                        )

    parser.add_argument('--save_dir',
                        type= str,
                        default= "checkpoint.pth",
                        help='Enter a directory to save the checkpoint file'
                        )

    parser.add_argument('--gpu',
                        action='store_true',
                        default=True,
                        help='Set a switch to true to use the GPU'
                        )

    input = parser.parse_args()

    return input


### Function for data preparation..

def data_loaders(data_dir, train_dir, valid_dir, test_dir, batch_size):
    """
    Function takes 5 arguments and use torchvision's transforms, and
    datasets modules. it takes data directory, load and prepare images for training and predictions.
    Command Line Arguments:
    1. Image directory as data_dir to load the main file.
    2. Train file directory as train_dir to load the training prepare, data.
    3. Validation file directory as valid_dir to load and prepare the valid
    data.
    4. Test file directory as test_dir to load and prepare the Test data.
    5. batch size as batch_size to specify the siza of training data.
    Function returns train_loader, valid_loader, test_loader, train_set
    can be used for training and inference later on.
    """

    train = transforms.Compose([transforms.RandomRotation(35),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
            ])

    train_set = datasets.ImageFolder(data_dir + train_dir, transform= train)
    train_loader = torch.utils.data.DataLoader(train_set,
                                batch_size=batch_size,
                                shuffle=True)

    valid = transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
                                ])

    valid_set = datasets.ImageFolder(data_dir + valid_dir, transform= valid)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size)

    test_set = datasets.ImageFolder(data_dir + test_dir, transform= valid)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, train_set


### Fuction that checks gpu..

def check_gpu(gpu):
    """
    Fuction takes one argument as boolean and provides support for gpu or cpu selection and print out the current device being used.
    Command Line Arguments:
    1. GPU as True value that enables GPU support and use cuda for calculation, and False to enable CPU.
    Function returns device and print out the device being used, either for trianing or inference.
    """
    # If gpu is True gpu is enabled and print out "\nGPU is availabe...". if the gpu didn't exist device switchs to cpu and print out "\nDevice didn't find GPU, using CPU instead"
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("\nGPU is availabe...")
        else:
             print("\nDevice didn't find GPU, using CPU instead")
    else:
        print("\nCPU is availabe...")
        return torch.device("cpu")

    return device

### Fuction that initiates complete transfer DNN...

def initiate_DNN(model, arch, hidden_units_1, hidden_units_2,
                learning_rate, device):
    """
    Fuction takes 6 arguments and initiates deep neural network, takes the downloaded torchvision model, freeze parameters, initiates new classifier with the required output units, criterion loss is defined as nn.NLLLoss(),
    and adam optimizer is defined.
    Command Line Arguments:
    1. torchvision model as model.
    2. model architechture as arch.
    3. hidden units 1 as hidden_units_1.
    4. hidden units 2 as hidden_units_2.
    4. learning rate as learning_rate for optimizer step.
    5. device as device being used.
    Fuction returns model, classifier, criterion, and optimizer.
    """

    for param in model.parameters():
        param.requires_grad = False

    if arch[:3] in ["vgg", "den", "ale", "squ", "mob"]:
        input_features = model.classifier[0].in_features
    else:
        input_features = model.fc.in_features

    classifier = nn.Sequential(OrderedDict([
                        ('fc1', nn.Linear(input_features, hidden_units_1)),
                        ('ReLu1', nn.ReLU()),
                        ('Dropout1', nn.Dropout(p=0.5)),
                        ('fc2', nn.Linear(hidden_units_1, hidden_units_2)),
                        ('ReLu1', nn.ReLU()),
                        ('fc3', nn.Linear(hidden_units_2, 102)),
                        ('output', nn.LogSoftmax(dim=1))
                        ]))

    if arch[:3] in ["vgg", "den", "ale", "squ", "mob"]:
        model.classifier = classifier
    else:
        model.fc = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    model.to(device);

    return model, classifier, criterion, optimizer

### Validation function...

def Validation(model, valid_loader, criterion, device):
    """
    Function takes 4 arguments and calculates the loss on Validation data.
    Function Prints out the valid loos and accuracy for trained model.
    Command Line Arguments:
    1. Model
    2. Validation loader as valid_loader to load the valid data.
    3. criterion loss to calculate the Loss
    4. device being used.
    Function returns validation loss, and accuracy.
    """

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
    """
    Fuction takes 7 arguments and train the classifier model been specified.
    This function train and print out running loss, validation loss, and accuracy.
    Command Line Arguments:
    1. Downloaded model.
    2. Train data as train_loader.
    3. Valid data as valid_loader.
    4. Number of epochs.
    5. Criterion loss.
    6. Optimizer step.
    7. Device.
    Function returns traind model.
    """

    steps = 0
    running_loss = 0
    print_every = 10

    print("\nStart training... Number of epochs: %s" % (epochs) )

    for epoch in range(epochs):
        print("Epoch: {}/{}".format(epoch+1, epochs))

        for images, labels in train_loader:
            steps += 1
            model.train()

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


    return model

### Function calculate the accuracy


def accuracy_check(model, test_loader, device):
    """
    Function takes 3 arguments and checks the accuracy on the test data,
    Function print out total accuracy.
    Command Line Arguments:
    1. Trained model.
    2. Test data as test_loader.
    3. Device.
    Function return None.
    """
    acc_tot = []

    with torch.no_grad():
        model.eval()

        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            logps = model.forward(images)

            outputs = torch.exp(logps)
            predictions, k_predictions = outputs.topk(1, dim=1)
            equals = k_predictions == labels.view(*k_predictions.shape)
            accuracy = (torch.mean(equals.type(torch.FloatTensor)).item())*100

            acc_tot.append(accuracy)

    print("\nTotal Test Accuracy: {:.2f}%".format(sum(acc_tot)/len(test_loader)))



def save_checkpoint(arch, epochs, save_dir, model, train_set, learning_rate,
            hidden_units_1, hidden_units_2):
    """
    Fuction takes 8 arguments and creates a dictionary check-point to save the model state_dict and other hyperparameters.
    Function has a save directory to save a new file.
    Command Line Arguments:
    1. Model architechture as arch.
    2. Number of epochs.
    3. Save directory as save_dir.
    4. Trained model as model.
    5. Training data as train_set to initiates class_to_idx.
    6. learning_rate
    7. hidden_units_1
    8.hidden_units_2
    Fuction returns None and print out "Model has saved successfully in %s"
    for save check up.
    """

    model.class_to_idx = train_set.class_to_idx

    checkpoint = {
                'architechture': arch,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx,
                'hidden_units_1': hidden_units_1,
                'hidden_units_2': hidden_units_2,
                'learning_rate': learning_rate
                }

    torch.save(checkpoint, save_dir)

    print("\nModel has saved successfully in %s" % (save_dir))



### main Fuction...

def main():

    start_time = time()

    input = input_args()

    train_dir = '/train'
    valid_dir = '/valid'
    test_dir =  '/test'


    train_loader, valid_loader, test_loader, train_set = data_loaders(
        input.data_dir, train_dir, valid_dir, test_dir, input.batch_size)

    device = check_gpu(gpu=True)

    model = models.__dict__[input.arch](pretrained=True)

    print("\nInitiating DNN parameters..... Model: %s, Hidden_units-1: %s, Hidden_units-2: %s, Learning_rate: %s" %
    (input.arch, input.hidden_units_1, input.hidden_units_2, input.learning_rate))

    model, classifier, criterion, optimizer = initiate_DNN(model, input.arch,
        input.hidden_units_1, input.hidden_units_2, input.learning_rate, device)


    model = train(model, train_loader, valid_loader, input.epochs, criterion,
                    optimizer, device)

    accuracy_check(model, test_loader, device)

    save_checkpoint(input.arch, input.epochs, input.save_dir, model, train_set,
            input.learning_rate, input.hidden_units_1, input.hidden_units_2)

    end_time = time()
    tot_time = end_time - start_time

    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" +
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )

if __name__ == "__main__":
    main()






##
