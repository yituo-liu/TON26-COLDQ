import torch
import torchvision
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
import random
import math
import pickle
import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import ScalarFormatter

def generate_WeightMatrix():
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

def quantize(x, s):
    norm_x = 0.005
    abs_x = np.abs(x)
    sign_x = np.sign(x)
    
    l = np.floor((abs_x / norm_x) * s).astype(int)
    prob = 1 - ((abs_x / norm_x) * s - l)
    
    random_vals = np.random.rand(*x.shape)
    quantized_x = np.where(random_vals < prob, l, l + 1)
    
    xtnhat = norm_x * sign_x * quantized_x / s
    
    return xtnhat, sign_x *  quantized_x

def conditional_entropy_numpy(param1, param2, b):
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

def COLDQ(N, total_step, D, d, C, Data, Label):
    x_max = 0.005
    constraint_summary = []
    epsilon = 8
    xtn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    ytn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    xtn_quantized = np.zeros((N, d * C))

    vq = epsilon * np.ones((N))
    constraint_upper = 1e-6
    current_ent_x = np.zeros((N, d * C))
    last_ent_x = np.zeros((N, d * C))

    Ent_Summary = []
    for step in range(1, total_step):
        alpha = 1e5
        GD_tnc_f = np.zeros((N, d * C))
        for n in range(N):
            for i in range(D):
                d_tni = Data[step - 1][n][i]
                b_tni = Label[step - 1][n][i]
                p_tni = np.zeros(C)
                for k in range(C):
                    idx = slice(d * k, d * (k + 1))
                    p_tni[k] = np.exp(d_tni @ xtn[step - 1][n][idx])
                hsum_tn = np.sum(p_tni)

                for c in range(0, C):
                    idx = slice(d * c, d * (c + 1))
                    GD_tnc_f[n][idx] -= (1 / D) * ((b_tni == c) - p_tni[c] / hsum_tn) * d_tni

            xtn[step][n] = ytn[step - 1][n] - 1 / (2 * alpha) * GD_tnc_f[n]
            xtn[step][n] = np.clip(xtn[step][n], -x_max, x_max)

            if np.linalg.norm(xtn[step][n] - xtn_quantized[n]) ** 2 > constraint_upper:
                xtn[step][n] = (2 * alpha * ytn[step - 1][n] - GD_tnc_f[n] + 2 * vq[n] * xtn_quantized[n]) / (2 * alpha + 2 * vq[n])
                xtn[step][n] = np.clip(xtn[step][n], -x_max, x_max)

        time_entropy = 0.0
        time_constraint = 0.0
        for n in range(N):
            last_ent_x[n] = current_ent_x[n]
            xtn_quantized[n], current_ent_x[n] = quantize(xtn[step][n], 7)

            Contraint = max(0, np.linalg.norm(xtn[step][n] - xtn_quantized[n]) ** 2 - constraint_upper)
            time_constraint += Contraint
            eta = 1 / np.power(step + 1, 2)
            gamma = epsilon * np.power(step + 1, 2)
            vq[n] = max(gamma, (1 - eta) * vq[n] + Contraint)
            
            time_entropy += conditional_entropy_numpy(torch.tensor(last_ent_x[n]), torch.tensor(current_ent_x[n]), 3)
        constraint_summary.append(time_constraint / N)
        Ent_Summary.append(time_entropy)
            
        ytn[step] = np.dot(WeightMatrix[step], xtn_quantized)
        if step % 100 == 0:
            print(f'COLDQ, t:{step}')

    Loss_Summary = np.zeros(total_step)
    for step in range(total_step):
        loss = 0
        for n in range(N):
            for i in range(D):
                for loss_n in range(N):
                    d_tni = Data[step][loss_n][i]
                    b_tni = int(Label[step][loss_n][i])
                    h_tni = np.zeros(C)
                    for k in range(C):
                        idx = slice(d * k, d * (k + 1))
                        h_tni[k] = np.exp(d_tni @ xtn[step][n][idx])
                    hsum_tn = np.sum(h_tni)
                    idx_2 = slice(d * (b_tni), d * (b_tni + 1))
                    loss -= np.log(np.exp(d_tni @ xtn[step][n][idx_2]) / hsum_tn) / N
        
        if step == 0:
            Loss_Summary[step] = loss / (N * D)
        else:
            Loss_Summary[step] = (Loss_Summary[step - 1]* step + loss / (N * D)) / (step + 1)
            print(f'Loss: {Loss_Summary[step]}, t:{step + 1}')

    # Test Accuracy
    Accuracy = []
    for t in range(total_step):
        wrong = 0
        for n in range(N):
            for i in range(1000):
                d_i = TestDataLabel[i, 1:785]
                h_ti = np.zeros(10)
                for k in range(10):
                    idx = slice(d * k, d * (k + 1))
                    h_ti[k] = np.exp(d_i @ xtn[t][n][idx])
                hsum_ti = np.sum(h_ti)
                TorF = (h_ti / hsum_ti).argmax()
                if TestDataLabel[i, 0] != TorF:
                    wrong += 1
        At = (1 - wrong / 1000 / N) * 100
        if t == 0:
            Accuracy.append(At)
        else:
            Accuracy.append((Accuracy[t - 1] * t + At) / (t + 1))
        print(f'Accuracy: {Accuracy[t]}, t:{t + 1}')
    return Loss_Summary, constraint_summary, Accuracy, Ent_Summary

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Random Seed
set_seed(42)

total_step = 500
N = 20
D = 32 # batch size
d = 784 # data dimension
C = 10 # number of classes


# Uncomment the following code to regenerate the weight matrix
# WeightMatrix = []
# rho = 0.1
# for t in range(total_step):
#     WeightMatrix.append(generate_WeightMatrix())


# Load the data from pickle files
script_dir = os.path.dirname(os.path.abspath(__file__))

data_file = os.path.join(script_dir, 'traindata.pkl')
label_file = os.path.join(script_dir, 'trainlabel.pkl')
test_file = os.path.join(script_dir, 'testdata&label.pkl')
weight_file = os.path.join(script_dir, 'WeightMatrix.pkl')

with open(data_file, 'rb') as f:
    Data = pickle.load(f)

with open(label_file, 'rb') as f:
    Label = pickle.load(f)

with open(test_file, 'rb') as f:
    TestDataLabel = pickle.load(f)

with open(weight_file, 'rb') as f:
    WeightMatrix = pickle.load(f)

# Simulation
Loss_COLDQ, Cons_COLDQ, Acc_COLDQ, Ent_COLDQ = COLDQ(N, total_step, D, d, C, Data, Label)



# Figure Plot
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

Ent_COLDQ_adjusted = adjust_entropy(Ent_COLDQ, 7840)

Cons_COLDQ = cumulative_average(Cons_COLDQ)

plt.figure(figsize=(20, 12))

plt.subplot(2, 2, 1)
plt.plot(Acc_COLDQ[1:500], label="COLDQ", color="red", linewidth=2)
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

plt.subplot(2, 2, 2)
plt.plot(Loss_COLDQ[1:500], label="COLDQ", color="red", linewidth=2)
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel(r"Training loss $\bar{f}(T)$", fontsize=24, fontdict={'fontname': 'Times New Roman'})
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

plt.subplot(2, 2, 3)
plt.plot(Ent_COLDQ_adjusted[1:500], label="COLDQ", color="red", linewidth=2)
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


plt.subplot(2, 2, 4)
plt.plot(Cons_COLDQ[1:500], label="COLDQ", color="red", linewidth=2)
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel(r"Hard Constraint Violation $\bar{g}(T)$", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.grid(True)
plt.legend(loc=(0.68, 0.15), prop={'family': 'Times New Roman', 'size': 20})
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

plt.subplots_adjust(wspace=1)

plt.tight_layout()
plt.show()
