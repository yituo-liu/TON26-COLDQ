import torch
import torch.nn as nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, ReLU, Softmax, CrossEntropyLoss
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torchvision.models import alexnet
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
import random
import pickle
from tqdm import tqdm, trange
from collections import defaultdict
from scipy.io import loadmat
from matplotlib.ticker import ScalarFormatter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset_tranform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=dataset_tranform)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=dataset_tranform)


def preload_data(dataset):
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    data, targets = next(iter(loader))
    return data.to(device), targets.to(device)

train_images_gpu, train_targets_gpu = preload_data(train_set)
test_images_gpu, test_targets_gpu = preload_data(test_set)

test_subset_indices = list(range(1000))
test_images_subset = test_images_gpu[test_subset_indices]
test_targets_subset = test_targets_gpu[test_subset_indices]


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x))) 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

class CifarResNet20(nn.Module):
    def __init__(self, num_classes=10):
        super(CifarResNet20, self).__init__()
        self.inplanes = 16
        block = BasicBlock
        layers = [3, 3, 3]

        self.conv1 = conv3x3(3, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet20(num_classes=10):

    return CifarResNet20(num_classes=num_classes)


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


initial_model = resnet20(num_classes=10)
state_dict = torch.load("resnet20_cifar10.pth", map_location='cpu')
initial_model.load_state_dict(state_dict, strict=True)

T = 500
N = 20

# Initialize Non-iid data indices for each device
label_indices = defaultdict(list)
for idx, (_, label) in enumerate(train_set):
    label_indices[label].append(idx)

main_labels = np.repeat(np.arange(10), 2)
np.random.shuffle(main_labels)
device_main_labels = {i: main_labels[i] for i in range(N)}

ind_n = {t: {n: [] for n in range(N)} for t in range(T)}
for t in range(T):
    if (t + 1) % 100 == 0:
        print(f"Training Data Prepare {t}")
    for i in range(N):
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
for t in range(T):
    weight_matrix.append(generate_WeightMatrix(N, 0.1))

def create_model(model):
    for name, param in model.named_parameters():
        param.requires_grad = False

    for name, param in model.named_parameters():
        if ("layer3.0.conv2" in name):
            param.requires_grad = True

    for name, module in model.named_modules():
        if ("layer3.0.conv2" in name):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    return model

local_models = [copy.deepcopy(initial_model).to(device) for i in range(N)]
local_models = [create_model(model) for model in local_models]

models_agg = [
    {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad} 
    for m in local_models
]

models_ent = [
    {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad} 
    for m in local_models
]

models_quantized = [
    {n: p.data.clone() for n, p in m.named_parameters() if p.requires_grad} 
    for m in local_models
]

loss = CrossEntropyLoss()
loss.to(device)

local_optimizers = []

for i in range(N):
    optimizer = torch.optim.SGD(local_models[i].parameters(), lr=0.01)
    local_optimizers.append(optimizer)

plot_accuracy = []
plot_loss = []

constraint_summary = torch.zeros(T).to(device)
Entropy = torch.zeros(T).to(device)
constraint_upper = 0.01

vq = torch.zeros(N).to(device)
beta = torch.tensor(1e-2).to(device)
V = torch.tensor(50).to(device)
Lambda = torch.tensor(1 / (2 * torch.sqrt(torch.tensor(500.0)))).to(device)
gtn = torch.zeros(N).to(device)

for t in range(T):
    transmit_power = 0
    weights_local = []
    loss_local = []

    for i in range(N):
        local_models[i].train()
        indices = ind_n[t][i]
        images = []
        targets = []
        sampled_images = train_images_gpu[indices]
        sampled_targets = train_targets_gpu[indices]

        outputs = local_models[i](sampled_images)
        result_loss = loss(outputs, sampled_targets)
        loss_local.append(result_loss.item())

        local_optimizers[i].zero_grad()
        result_loss.backward()
        with torch.no_grad():
            for param_tensor, param in local_models[i].named_parameters():
                if param.requires_grad and param.grad is not None:
                    param.data.copy_(models_agg[i][param_tensor] - 
                    0.01 * (param.grad * V * beta + Lambda * torch.exp(Lambda * vq[i]) * beta * gtn[i]))
                    param.data = torch.clamp(param.data, min=-0.1, max=0.1)
    
    time_constraint = torch.tensor(0.0).to(device)
    time_entropy = torch.tensor(0.0).to(device)
    for i in range(N):
        state_dict = local_models[i].state_dict()
        ent_previous_state_dict = copy.deepcopy(models_ent[i])
        ent_current_state_dict = copy.deepcopy(models_ent[i])
        for param_tensor, param in local_models[i].named_parameters():
            if param.requires_grad and param.grad is not None:
                if t == 0:
                    ent_previous_state_dict[param_tensor] = torch.zeros(param.shape).to(device)
                state_dict[param_tensor], ent_current_state_dict[param_tensor] = quantize(state_dict[param_tensor], 3)
                time_entropy += conditional_entropy_torch(ent_previous_state_dict[param_tensor], ent_current_state_dict[param_tensor], 2)
                models_quantized[i][param_tensor] = state_dict[param_tensor]
        models_ent[i] = ent_current_state_dict

        gradient_norm_sum = torch.tensor(0.0).to(device)
        with torch.no_grad():
            for param_tensor, param in local_models[i].named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradient_norm_sum += torch.norm(param.data - models_quantized[i][param_tensor]) ** 2

        Constraint = torch.max(torch.tensor(0.0).to(device), gradient_norm_sum - constraint_upper)
        time_constraint += Constraint
        vq[i] = vq[i] + Constraint * beta
        gtn[i] = Constraint
    constraint_summary[t] = time_constraint / N
    Entropy[t] = time_entropy
    current_weights = weight_matrix[t]
    temp_trained_states = copy.deepcopy(models_quantized)
    for i in range(N):
        neighbors = np.where(current_weights[i] > 0)[0]
        
        state_dict = models_agg[i]
        with torch.no_grad():
            for key in state_dict.keys():

                new_param = torch.zeros_like(state_dict[key])
                for j in neighbors:
                    w = current_weights[i][j]
                    new_param.add_(temp_trained_states[j][key], alpha=float(w))
                state_dict[key].copy_(new_param)

    if (t+1) % 1 == 0:
        total_correct = 0
        with torch.no_grad():
            for i in range(N):
                local_models[i].eval()
                outputs = local_models[i](test_images_subset)
                pred = outputs.argmax(dim=1)
                total_correct += (pred == test_targets_subset).sum().item()
        if t == 0:
            plot_accuracy.append(total_correct / 20000 * 100)
        else:
            plot_accuracy.append((plot_accuracy[t-1] * t + total_correct / 20000 * 100) / (t+1))
        print("t: {} Accuracy: {:.2f} Cons: {:.4f} Ent: {:.4f} Qt: {}".format(t+1, plot_accuracy[t], constraint_summary[t], Entropy[t], vq.mean().item()))

def adjust_entropy(entropy_data, num_params):
    adjusted_entropy = np.zeros(len(entropy_data))
    cumulative_sum = 0
    for i in range(len(entropy_data)):
        cumulative_sum += entropy_data[i] * num_params / 1e6
        adjusted_entropy[i] = cumulative_sum
    return adjusted_entropy

Ent_adjusted = adjust_entropy(torch.tensor(Entropy).cpu().numpy(), 36864)


# Plotting Results
plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(plot_accuracy[1:T], label="[28]", color="red", linewidth=2)
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
plt.plot(Ent_adjusted[1:T], label="[28]", color="red", linewidth=2)
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