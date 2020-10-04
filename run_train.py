
from train import *

import argparse
import torch
from torch import optim, nn
from torchvision import transforms, datasets, models
from collections import OrderedDict
import torch.nn.functional as F
from time import time



def main():

    start_time = time()

    input = input_args()

    train_dir = '/train'
    valid_dir = '/valid'
    test_dir =  '/test'


    train_loader, valid_loader, test_loader, train_set = data_loaders(
    input.data_dir, train_dir, valid_dir, test_dir)

    device = check_gpu()

    exec("model = models.{}(pretrained=True)".format(input.arch))

    model, classifier, criterion, optimizer = initiate_DNN(model, input.arch,
    input.hidden_units, input.learning_rate, device)


    model = train(model, train_loader, valid_loader, input.epochs, criterion,
     optimizer, device)

    accuracy_check(model, test_loader, device)

    save_checkpoint(input.arch, input.epochs, input.save_dir, model, train_set)

    end_time = time()
    tot_time = end_time - start_time

    print("\nTotal Elapsed Runtime:", str( int( (tot_time / 3600) ) ) + ":" +
          str( int(  ( (tot_time % 3600) / 60 )  ) ) + ":" +
          str( int(  ( (tot_time % 3600) % 60 ) ) ) )

if __name__ == "__main__":
    main()
