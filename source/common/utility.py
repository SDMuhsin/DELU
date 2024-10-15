import torch.nn as nn
from common.DELU import *
from common.DELU import DELU,ADELU,TDELU,ATDELU,FADELU,SGELU,SWLU,RGELU,SRGELU,SEGELU

def get_activation_by_name(activation_name,a=1.0,b=1.0,c=1.0,d=1.0):
    activation_map = {
        'ReLU': nn.ReLU(),
        'LeakyReLU': nn.LeakyReLU(),
        'ELU': nn.ELU(),
        'SELU': nn.SELU(),
        'GELU': nn.GELU(),
        'Tanh': nn.Tanh(),
        'Sigmoid': nn.Sigmoid(),
        'Hardswish': nn.Hardswish(),
        'Mish': nn.Mish(),
        'SiLU': nn.SiLU(),
        'Softplus': nn.Softplus(),
        'Softsign': nn.Softsign(),
        'Hardshrink': nn.Hardshrink(),
        'Softshrink': nn.Softshrink(),
        'Tanhshrink': nn.Tanhshrink(),
        'PReLU': nn.PReLU(),
        'RReLU': nn.RReLU(),
        'CELU': nn.CELU(),
        'Hardtanh': nn.Hardtanh(),
        'DELU' : DELU(a,b),
        'ADELU' : ADELU(a,b),
        'TDELU' : TDELU(),
        'ATDELU' : ATDELU(),
        'FADELU' : FADELU(a,b,c,d),
        'SGELU'  : SGELU(),
        'SWLU'   : SWLU(),
        'RGELU'  : RGELU(),
        'SRGELU' : SRGELU(),
        'SEGELU' : SEGELU(),
        'HGELU'  : HGELU(),
        'SQGELU' : SQGELU(),
        'LOGGELU' : LOGGELU()
    }
    print(f"a = {a}, d = {d}") 
    if activation_name in activation_map:
        return activation_map[activation_name]
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")
def replace_activations(module,old_activation,new_activation):
    """
    Recursively replaces all ReLU activations in the model with SmoothExponentialLinear (SEL).
    Args:
        module (nn.Module): The PyTorch model or module where replacements are needed.
    """
    for name, child in module.named_children():
        if isinstance(child, old_activation):
            # If the child is old_activation, replace with new activation
            print(f"Replacing {name}")
            setattr(module, name, new_activation)
        else:
            # Recursively apply to child modules
            replace_activations(child,old_activation,new_activation)

from torchvision import datasets, transforms, models


def get_model(model_name, dataset):

    num_classes = {
        'mnist': 10,
        'fmnist': 10,
        'cifar10': 10,
        'cifar100': 100,
        'svhn': 10,
        'stl10': 10,
        'emnist': 47,
        'kmnist': 10
    }

    if dataset not in num_classes:
        raise ValueError(f"Unsupported dataset: {dataset}")

    output_dim = num_classes[dataset]
    input_channels = 1 if dataset in ['mnist', 'fmnist', 'emnist', 'kmnist'] else 3

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        num_ftrs = model.fc.in_features

        if(dataset in ['kmnist','emnist']):
             model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(num_ftrs, output_dim)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=False)
        
        # Modify the first convolutional layer to accept different input channels
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=11, stride=4, padding=2)
        
        # Modify the classifier to output the correct number of classes
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, output_dim)
        
        # Adjust the model for smaller input sizes if necessary
        if dataset in ['mnist', 'fmnist', 'emnist', 'kmnist', 'cifar10', 'cifar100']:
            model.features[0].stride = (1, 1)
            model.features[0].padding = (2, 2)
            model.features[3].kernel_size = (3, 3)
            model.features[3].stride = (1, 1)
            model.features[3].padding = (1, 1)
            
            # Remove the second max pooling layer
            new_features = nn.Sequential(*list(model.features.children())[:-3])
            model.features = new_features
            
            # Adjust the classifier for the new feature map size
            model.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, output_dim),
            )
    elif model_name == 'resnet34':

        model = models.resnet34(pretrained=False)
        num_ftrs = model.fc.in_features
        if(dataset in ['kmnist','emnist']):
             model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(num_ftrs, output_dim)

    elif model_name == 'resnet50':

        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        if(dataset in ['kmnist','emnist']):
             model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(num_ftrs, output_dim)


    elif model_name == 'shufflenet':

        model = models.shufflenet_v2_x0_5(pretrained=False)
        if(dataset in ['kmnist','emnist']):
            model.conv1[0] = nn.Conv2d(input_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, output_dim)

    elif model_name == 'vgg16':

        model = models.vgg16(pretrained=False)
        model.features[0] = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, output_dim)

    elif model_name == 'densenet121':

        model = models.densenet121(pretrained=False)
        model.features.conv0 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)

    elif model_name == 'smallnet':

        model = SmallNet()
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, output_dim)

    else:

        raise ValueError(f"Unsupported model: {model_name}")



    return model

