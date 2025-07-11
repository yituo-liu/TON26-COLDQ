import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Softmax, CrossEntropyLoss
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.models import alexnet
import numpy as np
import copy
import time
from torchsummary import summary
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm, trange
import pickle
import random
from matplotlib.ticker import ScalarFormatter
from collections import defaultdict

# Data Preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Load CIFAR-10 dataset
trainset = CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = CIFAR10(root='./data', train=False, download=True, transform=transform)

# Model Aggregation
def FedAvg(mods, weight_mat):
    model_params = [copy.deepcopy(model.state_dict()) for model in mods]

    for i in range(len(mods)):

        weights_avg_list = {key: torch.zeros_like(value) for key, value in model_params[0].items() if key in ['classifier.6.weight', 'classifier.6.bias']}
        for key in weights_avg_list.keys():
            for j in range(len(mods)):
                if weight_mat[i, j] > 0:
                    weights_avg_list[key] += weight_mat[i, j] * model_params[j][key]

        classifier_state_dict = {k.split('classifier.6.')[1]: weights_avg_list[k] for k in weights_avg_list}
        mods[i].classifier[6].load_state_dict(classifier_state_dict)

    return mods


def create_model():
    model = torchvision.models.alexnet(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 10)

    nn.init.kaiming_normal_(model.classifier[6].weight)
    nn.init.constant_(model.classifier[6].bias, 0)

    for param in model.classifier[6].parameters():
        param.requires_grad = True

    return model

def train(device, models, criterion, optimizers, weight_matrix, models_agg):
    loss_list = torch.zeros(time_slot).to(device)
    plot_accuracy = torch.zeros(time_slot).to(device)
    pbar = tqdm(range(time_slot), 
                desc='Training Progress', 
                ncols=100,
                unit='Slot')
    
    for i in range(num_devices):
        with torch.no_grad():
            for param_tensor, param in models[i].named_parameters():
                if param.requires_grad and param.grad is not None:
                    max_params = torch.max(torch.abs(param.data))
    for t in pbar:
        running_loss = 0.0
        time_accuracy = 0
        for i in range(len(models)):
            models[i].train()
            indices = ind_n[t][i]
            images = []
            targets = []
            for idx in indices:
                img, tar = trainset[idx]
                images.append(img)
                targets.append(tar)

            images = torch.stack(images)
            targets = torch.tensor(targets)

            sampled_images = images.to(device)
            sampled_targets = targets.to(device)

            optimizers[i].zero_grad()
            outputs = models[i](sampled_images)
            loss = criterion(outputs, sampled_targets)
            loss.backward()
            with torch.no_grad():
                for param_tensor, param in models[i].named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data = models_agg[i].state_dict()[param_tensor] - param.grad * 0.01
                        param.data = torch.clamp(param.data, min=-0.1, max=0.1)

            running_loss += loss.item()
            models[i].eval()
            with torch.no_grad():
                for data in test_dataloader:
                    imgs, targets = data
                    imgs = imgs.to(device)
                    targets = targets.to(device)
                    outputs = models[i](imgs)
                    accuracy = (outputs.argmax(1) == targets).sum()
                    time_accuracy += accuracy
        plot_accuracy[t] = torch.Tensor.cpu(time_accuracy / len(test_subset) / num_devices) * 100            

        time_loss = running_loss / len(models)
        loss_list[t] = time_loss
        if (t + 1) % 1 == 0:
            pbar.write(f'Idealized DL {t+1}, Loss: {time_loss:.2f}, Acc: {plot_accuracy[t]:.2f}%')

        pbar.set_postfix()

        with torch.no_grad():
            models_agg = FedAvg(models, weight_matrix[t])
    
    return loss_list, plot_accuracy

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Random Seed
set_seed(42)

# Initialize models and optimizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_devices = 20

models = [create_model().to(device) for _ in range(num_devices)]
models_agg = copy.deepcopy(models)
optimizers = [torch.optim.SGD(model.parameters(), lr=0.01) for model in models]
criterion = nn.CrossEntropyLoss()

time_slot = 500

# Initialize Non-iid data indices for each device
label_indices = defaultdict(list)
for idx, (_, label) in enumerate(trainset):
    label_indices[label].append(idx)

main_labels = np.repeat(np.arange(10), 2)
np.random.shuffle(main_labels)
device_main_labels = {i: main_labels[i] for i in range(num_devices)}

ind_n = {t: {n: [] for n in range(num_devices)} for t in range(time_slot)}
for t in range(time_slot):
    if (t + 1) % 100 == 0:
        print(f"Training Data Prepare {t}")
    for i in range(num_devices):
        selected_indices = []
        main_label = device_main_labels[i]
        
        num_main_samples = int(0.2 * 32)
        main_samples = np.random.choice(label_indices[main_label], num_main_samples, replace=False)
        selected_indices.extend(main_samples)
        
        num_other_samples = 32 - num_main_samples
        other_labels = [label for label in np.arange(10) if label != main_label]
        other_samples = np.random.choice(np.concatenate([label_indices[label] for label in other_labels]), num_other_samples, replace=False)
        selected_indices.extend(other_samples)
        
        ind_n[t][i] = selected_indices


subset_indices = list(range(5000))
test_subset = Subset(testset, subset_indices)
test_dataloader = DataLoader(test_subset, batch_size=64, drop_last=False)

# Weight Matrix Generation
def generate_WeightMatrix(N, rho):
    P = np.zeros((N, N))
    for i in range(N - 1):
        P[i + 1, i] = 1 / N
        P[i, i + 1] = 1 / N

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() <= rho:
                P[i, j] = 1 / N
                P[j, i] = 1 / N
        P[i, i] = 1 - np.sum(P[i, :])
    return P
weight_matrix = []
for t in range(time_slot):
    weight_matrix.append(generate_WeightMatrix(num_devices, 0.1))

# Training
Loss, acc = train(device, models, criterion, optimizers, weight_matrix, models_agg)

acc = acc.cpu().clone().detach()

def cumulative_average(data):
    cumulative_sum = 0
    cumulative_avg = np.zeros(len(data))
    for i in range(len(data)):
        cumulative_sum += data[i]
        cumulative_avg[i] = cumulative_sum / (i + 1)
    return cumulative_avg

acc = cumulative_average(acc.numpy())


# Plotting Results
plt.figure(figsize=(16, 8))

plt.plot(acc[1:time_slot], label="Idealized DL", color="black", linestyle=':', linewidth=2)
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel(r"Test accuracy $\bar{A}(T)$ (%)", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.grid(True)
plt.legend(prop={'family': 'Times New Roman', 'size': 20})
ax1 = plt.gca()
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')

plt.tight_layout()
plt.show()

# Save results
# with open('Nonconvex_IdealizedDL_acc.pkl', 'wb') as f:
#     pickle.dump(acc, f)
