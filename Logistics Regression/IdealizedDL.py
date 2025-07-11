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

def ErrorFree(N, total_step, D, d, C, Data, Label):
    x_max = 0.005
    xtn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}
    ytn = {t: np.zeros((N, d * C)) for t in range(0, total_step)}

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

        ytn[step] = np.dot(WeightMatrix[step], xtn[step])
        if step % 100 == 0:
            print(f'ErrorFree, t:{step}')

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

    return Loss_Summary, Accuracy

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
Loss_Ideal, Acc_Ideal = ErrorFree(N, total_step, D, d, C, Data, Label)



# Figure Plot
plt.figure(figsize=(20, 12))

plt.subplot(1, 2, 1)
plt.plot(Acc_Ideal[1:500], label="Error-free DL", color="black", linestyle=':', linewidth=2)
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
plt.plot(Loss_Ideal[1:500], label="Error-free DL", color="black", linestyle=':', linewidth=2)
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

plt.tight_layout()
plt.show()
