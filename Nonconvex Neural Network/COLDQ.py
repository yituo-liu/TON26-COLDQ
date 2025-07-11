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
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm, trange
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

def quantize(x, s):
    norm_x = torch.tensor(0.1).to(device)
    abs_x = torch.abs(x)
    sign_x = torch.sign(x)
    
    l = torch.floor((abs_x / norm_x) * s).int()
    prob = 1 - ((abs_x / norm_x) * s - l)
    
    random_vals = torch.rand_like(x)
    quantized_x = sign_x * torch.where(random_vals < prob, l, l + 1)
    
    xtnhat = norm_x * quantized_x / s
    
    return xtnhat, quantized_x

def conditional_entropy_torch(param1, param2, b):
    lmax = torch.tensor(2**b - 1).to(param1.device)
    edge = torch.arange(-lmax, lmax + 1, device=param1.device)
    levels = torch.tensor(2**(b + 1) - 2).to(param1.device)

    x_bin = torch.floor(((param1 - edge[0]) / (edge[-1] - edge[0]) * levels)).long()
    y_bin = torch.floor(((param2 - edge[0]) / (edge[-1] - edge[0]) * levels)).long()

    mask = (x_bin >= 0) & (x_bin < levels) & (y_bin >= 0) & (y_bin < levels)
    x_bin = x_bin[mask]
    y_bin = y_bin[mask]
   
    joint_hist = torch.zeros((levels, levels), device=param1.device).flatten()
    indices = x_bin * levels + y_bin
    joint_hist.index_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))
    joint_hist = joint_hist.view(levels, levels)

    joint_prob = joint_hist / param1.numel()

    hist1 = torch.histc(param1, bins=levels, min=edge[0], max=edge[-1]).to(param1.device)

    prob1 = hist1 / param1.numel()

    cond_prob = joint_prob / prob1.unsqueeze(1).expand_as(joint_prob)

    mask = (joint_prob > 0) & (cond_prob > 0) & torch.isfinite(joint_prob) & torch.isfinite(cond_prob)

    cond_entropy = -torch.sum(joint_prob[mask] * torch.log2(cond_prob[mask]))

    return cond_entropy.item()

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

def COLDQ(device, models, criterion, optimizers, weight_matrix, models_quantized, models_agg):
    constraint_summary = torch.zeros(time_slot).to(device)
    loss_list = torch.zeros(time_slot).to(device)
    plot_accuracy = torch.zeros(time_slot).to(device)
    Entropy = np.zeros(time_slot)
    pbar = tqdm(range(time_slot), 
                desc='Training Progress', 
                ncols=100,
                unit='Slot')
    epsilon = 0.02
    vq = epsilon * torch.ones(num_devices).to(device)
    constraint_upper = 1e-4
    for t in pbar:
        running_loss = 0.0
        alpha = 50
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

            outputs = models[i](sampled_images)
            loss = criterion(outputs, sampled_targets)
            optimizers[i].zero_grad()
            loss.backward()

            running_loss += loss.item()

            gradient_norm_sum = torch.tensor(0.0).to(device)
            with torch.no_grad():
                for param_tensor, param in models[i].named_parameters():
                    if param.requires_grad and param.grad is not None:
                        param.data = models_agg[i].state_dict()[param_tensor] - param.grad / (2 * alpha)
                        param.data = torch.clamp(param.data, min=-0.1, max=0.1)
                        gradient_norm_sum += torch.norm(param.data - models_quantized[i].state_dict()[param_tensor]) ** 2

            if gradient_norm_sum > constraint_upper:
                with torch.no_grad():
                    for param_tensor, param in models[i].named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param.data = (2 * alpha * models_agg[i].state_dict()[param_tensor] - param.grad
                                           + 2 * vq[i] * models_quantized[i].state_dict()[param_tensor]) / (2 * alpha + 2 * vq[i])
                            param.data = torch.clamp(param.data, min=-0.1, max=0.1)

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

        time_constraint = torch.tensor(0.0).to(device)
        time_entropy = 0.0
        for i in range(len(models)):
            state_dict = models[i].state_dict()
            ent_previous_state_dict = copy.deepcopy(models_ent[i].state_dict())
            ent_current_state_dict = copy.deepcopy(models_ent[i].state_dict())

            for param_tensor, param in models[i].named_parameters():
                if param.requires_grad and param.grad is not None:
                    if t == 0:
                        ent_previous_state_dict[param_tensor] = torch.zeros(param.shape).to(device)
                    state_dict[param_tensor], ent_current_state_dict[param_tensor] = quantize(state_dict[param_tensor], 7)
                    time_entropy += conditional_entropy_torch(ent_previous_state_dict[param_tensor], ent_current_state_dict[param_tensor], 3)

            models_quantized[i].load_state_dict(state_dict)
            models_ent[i].load_state_dict(ent_current_state_dict)

            gradient_norm_sum = torch.tensor(0.0).to(device)
            with torch.no_grad():
                for param_tensor, param in models[i].named_parameters():
                    if param.requires_grad and param.grad is not None:
                        gradient_norm_sum += torch.norm(param.data - models_quantized[i].state_dict()[param_tensor]) ** 2

            Constraint = torch.max(torch.tensor(0.0).to(device), gradient_norm_sum - constraint_upper)
            time_constraint += Constraint
            eta = 1 / torch.pow(torch.tensor(t + 1.0).to(device), 2)
            gamma = epsilon * torch.pow(torch.tensor(t + 1.0).to(device), 2)
            vq[i] = torch.max(gamma, (1 - eta) * vq[i] + Constraint)
        constraint_summary[t] = time_constraint / num_devices

        time_loss = running_loss / len(models)
        loss_list[t] = time_loss
        Entropy[t] = time_entropy
        if (t + 1) % 1 == 0:
            pbar.write(f'COLDQ {t+1}, Loss: {time_loss:.2f}, Acc: {plot_accuracy[t]:.2f}%, Cons: {constraint_summary[t]:.2f}, Ent: {Entropy[t]:.4f}')

        pbar.set_postfix()
        
        with torch.no_grad():
            models_agg = FedAvg(models_quantized, weight_matrix[t])
    
    return loss_list, constraint_summary, plot_accuracy, Entropy


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
models_quantized = copy.deepcopy(models)
models_agg = copy.deepcopy(models)
models_ent = copy.deepcopy(models)

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
Loss_COLDQ, Cons_COLDQ, Acc_COLDQ, Ent_COLDQ = COLDQ(device, models, criterion, optimizers, weight_matrix, models_quantized, models_agg)


Acc_COLDQ = Acc_COLDQ.cpu().clone().detach()
Ent_COLDQ = torch.tensor(Ent_COLDQ).cpu()


def cumulative_average(data):
    cumulative_sum = 0
    cumulative_avg = np.zeros(len(data))
    for i in range(len(data)):
        cumulative_sum += data[i]
        cumulative_avg[i] = cumulative_sum / (i + 1)
    return cumulative_avg

def adjust_entropy(entropy_data, num_params):
    adjusted_entropy = np.zeros(len(entropy_data))
    cumulative_sum = 0
    for i in range(len(entropy_data)):
        cumulative_sum += entropy_data[i] * num_params / 1e6
        adjusted_entropy[i] = cumulative_sum
    return adjusted_entropy

Acc_COLDQ = cumulative_average(Acc_COLDQ.numpy())
Ent_COLDQ_adjusted = adjust_entropy(Ent_COLDQ.numpy(), 40970)


# Plotting Results
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(Acc_COLDQ[1:time_slot], label="COLDQ", color="red", linewidth=2)
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


plt.subplot(1, 2, 2)
plt.plot(Ent_COLDQ_adjusted[1:time_slot], label="COLDQ", color="red", linewidth=2)
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel(r"Transmitted bits $B(T)$ (Mb)", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.grid(True)
plt.legend(prop={'family': 'Times New Roman', 'size': 20})
ax2 = plt.gca()
ax2.tick_params(axis='x', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)
for label in ax2.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax2.get_yticklabels():
    label.set_fontname('Times New Roman')
ax2.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax2.yaxis.offsetText.set_fontsize(20)
ax2.yaxis.offsetText.set_fontname('Times New Roman')

plt.tight_layout()
plt.show()

# Save results
# with open('Nonconvex_COLDQ_Acc.pkl', 'wb') as f:
#     pickle.dump(Acc_COLDQ, f)

# with open('Nonconvex_COLDQ_Ent.pkl', 'wb') as f:
#     pickle.dump(Ent_COLDQ_adjusted, f)