import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import random
import itertools

import pyDOE2
from scipy.optimize import minimize
from skopt import gp_minimize

import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import datetime
class SimpleNN(nn.Module):
    def __init__(self, to_demo=True):
        super(SimpleNN, self).__init__()
        if to_demo:
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc1 = nn.Linear(32 * 8 * 8, 64)
            self.fc2 = nn.Linear(64, 10)
        else:
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.fc1 = nn.Linear(64 * 8 * 8, 128)
            self.fc2 = nn.Linear(128, 10)

        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

'''
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 8 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
'''
'''#This is originally work version
class SimpleNN(nn.Module,to_demo=True):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Change the input size from 64 * 16 * 16 to 64 * 8 * 8
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
'''
'''
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Changed the number of input channels from 1 to 3
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
'''
'''
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
'''
'''
def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
'''
def init_weights(model, init_values):
    for i, param in enumerate(model.parameters()):
        param.data.copy_(torch.from_numpy(init_values[i]))

        #param.data.copy_(torch.tensor(init_values[i]))


# SimpleNN and init_weights remain the same as before
def simulate_parallel_loss(model, init_values, train_loader, device, num_epochs=10):
    model.to(device)
    init_weights(model, init_values)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    loss_curve = []
    for epoch in tqdm(range(num_epochs), desc="Training"):#range(num_epochs):
        running_loss = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        loss_curve.append(running_loss / len(train_loader))
    return loss_curve

'''
def run_simulation(num_workers, init_values_list, train_loader, device, num_epochs=10, to_demo=True):
    losses = []
    for worker_id in range(num_workers):
        init_values = init_values_list[worker_id]
        loss_curve = simulate_parallel_loss(SimpleNN(to_demo=to_demo), init_values, train_loader, device, num_epochs)
        losses.append(loss_curve)
    return losses
'''
#originally work version
def run_simulation(num_workers,init_values, train_loader, device, num_epochs=10,to_demo=True):#original 2nd argument:sampling_method
    losses = []
    for worker_id in range(num_workers):
        #init_values = sampling_method()
        loss_curve = simulate_parallel_loss(SimpleNN(to_demo=to_demo), init_values, train_loader, device, num_epochs)
        losses.append(loss_curve)
    return losses

'''
def simulate_parallel_loss(model, init_values, train_loader, device):
    model.to(device)
    init_weights(model, init_values)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    data, target = next(iter(train_loader))
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    return loss.item()

def run_simulation(num_workers, sampling_method, train_loader, device):
    losses = []
    for worker_id in range(num_workers):
        init_values = sampling_method()
        loss = simulate_parallel_loss(SimpleNN(), init_values, train_loader, device)
        losses.append(loss)
    return losses
'''
'''
def run_simulation(num_workers, sampling_method, train_loader, device, num_epochs=10):
    losses = []
    for worker_id in range(num_workers):
        init_values = sampling_method()
        model = SimpleNN().to(device)
        init_weights(model, init_values)
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        loss_curve = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            loss_curve.append(running_loss / len(train_loader))
        losses.append(loss_curve)
    return losses
'''

def main(to_demo=True):
        # Prepare the dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    if to_demo:
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [1000, len(train_dataset) - 1000])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)#, num_workers=2
    NUM_WORKERS_= 4
    num_workers = NUM_WORKERS_
    
    num_epochs = 30 if to_demo else 50 #30
    # Define sampling methods
    def single_random_initialization():
        model = SimpleNN()
        init_values = [p.data.clone().numpy() for p in model.parameters()]# originally no .numpy()
        return init_values


    def uniform_sampling():
        return [np.random.uniform(-1, 1, p.numel()).reshape(p.shape) for p in SimpleNN().parameters()]
    
    def latin_hypercube_sampling():
        n_params = sum(p.numel() for p in SimpleNN().parameters())
        lhs_samples = pyDOE2.lhs(n_params, samples=num_workers, criterion='maximin')
        lhs_samples = lhs_samples * 2 - 1  # scale to [-1, 1]
        
        init_values_list = []
        for sample in lhs_samples:
            init_values = []
            start_index = 0
            for p in SimpleNN().parameters():
                end_index = start_index + p.numel()
                param_sample = sample[start_index:end_index].reshape(p.shape)
                init_values.append(torch.from_numpy(param_sample))
                start_index = end_index
            init_values_list.append(init_values)

        return init_values_list

    '''
    def latin_hypercube_sampling():
        n_params = sum(p.numel() for p in SimpleNN().parameters())
        lhs_samples = pyDOE2.lhs(n_params, samples=num_workers, criterion='maximin')
        lhs_samples = lhs_samples * 2 - 1  # scale to [-1, 1]
        
        init_values_list = []
        for sample in lhs_samples:
            init_values = []
            start_index = 0
            for p in SimpleNN().parameters():
                end_index = start_index + p.numel()
                param_sample = sample[start_index:end_index].reshape(p.shape)
                init_values.append(param_sample)
                start_index = end_index
            init_values_list.append(init_values)

        return init_values_list
    '''
    '''
    def latin_hypercube_sampling():
        n_params = sum(p.numel() for p in SimpleNN().parameters())
        lhs_samples = pyDOE2.lhs(n_params, samples=num_workers, criterion='maximin')
        lhs_samples = lhs_samples * 2 - 1  # scale to [-1, 1]
        
        init_values = []
        start_index = 0
        for p in SimpleNN().parameters():
            end_index = start_index + p.numel()
            sample = lhs_samples[:, start_index:end_index].reshape(p.shape)
            init_values.append(sample)
            start_index = end_index

        return init_values
    '''
    '''
    def latin_hypercube_sampling():
        n_params = sum(p.numel() for p in SimpleNN().parameters())
        lhs_samples = pyDOE2.lhs(n_params, samples=num_workers, criterion='maximin')
        lhs_samples = lhs_samples * 2 - 1  # scale to [-1, 1]
        return  [sample.reshape((1, -1)) for sample in lhs_samples]#[list(sample.reshape((1, -1))) for sample in lhs_samples]
    '''
    def adaptive_sampling():
        def loss_function(params):
            init_values = [param.reshape(p.shape) for param, p in zip(params, SimpleNN().parameters())]
            return simulate_parallel_loss(SimpleNN(), init_values, train_loader, device)

        bounds = [(-1, 1)] * sum(p.numel() for p in SimpleNN().parameters())
        result = minimize(loss_function, x0=np.zeros(len(bounds)), bounds=bounds, method='L-BFGS-B')
        best_params = result.x

        return [best_params.reshape(p.shape) for p in SimpleNN().parameters()]
    '''
    def adaptive_sampling():
        def loss_function(params):
            init_values = [param.reshape(p.shape) for param, p in zip(params, SimpleNN().parameters())]
            return simulate_parallel_loss(SimpleNN(), init_values, train_loader, device)

        bounds = [(-1, 1)] * sum(p.numel() for p in SimpleNN().parameters())
        result = minimize(loss_function, x0=np.zeros(len(bounds)), bounds=bounds, method='L-BFGS-B')
        best_params = result.x

        return [best_params.reshape(p.shape) for p in SimpleNN().parameters()]
    '''
    def bayesian_optimization():
        def loss_function(params):
            init_values = [param.reshape(p.shape) for param, p in zip(params, SimpleNN().parameters())]
            return simulate_parallel_loss(SimpleNN(), init_values, train_loader, device)

        bounds = [(-1, 1)] * sum(p.numel() for p in SimpleNN().parameters())
        result = gp_minimize(loss_function, bounds, n_calls=num_workers, n_random_starts=0, random_state=42)
        best_params = result.x

        return [best_params.reshape(p.shape) for p in SimpleNN().parameters()]

    methods = [
        ('Single Random Initialization', single_random_initialization),
        ('Uniform Sampling', uniform_sampling),
        #('Adaptive Sampling', adaptive_sampling),
        #('Bayesian Optimization', bayesian_optimization),
        #('LHS', latin_hypercube_sampling),
    ]

    losses = {}
    for method_name, method in methods:
        print(f"Running {method_name}...")
        if method_name=='Single Random Initialization':
            num_workers=1
        else:
            num_workers=NUM_WORKERS_
        init_values = method()
        loss_curve = run_simulation(num_workers, init_values, train_loader, device, num_epochs,to_demo)
        losses[method_name] = loss_curve

    print(losses)
    # Plot loss curves
    plt.figure(figsize=(12, 6))
    for method_name, loss_curve in losses.items():
        plt.plot(loss_curve, label=method_name)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves for Different Initialization Methods')
    plt.legend()
    plt.grid()
    plt.show()

    results_dir = './results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    plt.savefig(os.path.join(results_dir, datetime.date.today().strftime("%B %d, %Y") + 'loss_curves.png'))
    plt.show()


'''
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    num_workers = 8
    num_iterations = 10

    

    sampling_methods = [uniform_sampling]  # Add other sampling methods here
    sampling_names = ['Uniform Sampling']  # Add other sampling method names here

    for method, name in zip(sampling_methods, sampling_names):
        print(f"Simulating {name}")
        all_losses = []
        for _ in range(num_iterations):
            losses = run_simulation(num_workers, method, train_loader, device)
            all_losses.append(losses)
            print(f"Losses: {losses}")

        mean_loss = np.mean(all_losses)
        print(f"Mean loss for {name}: {mean_loss}")
'''
if __name__ == "__main__":
    main(to_demo=True)
