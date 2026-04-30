import random
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
import pickle
import time
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import ScalarFormatter

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Seed Setting
set_seed(44)

x_dim = 10
constraint_dim = 2
A_range = (0, 1)
b_range = (0, 1)
total_steps = 1000
N = 20  # Number of Devices
rho = 0.1  # Connection Probability

run_number = 20  # Number of Runs

def generate_H():
    return np.random.uniform(low=-1, high=1, size=(4, x_dim))

def generate_A():
    return np.random.uniform(low=A_range[0], high=A_range[1], size=(constraint_dim, x_dim))

def generate_b():
    return np.random.uniform(low=b_range[0], high=b_range[1], size=(constraint_dim, 1))

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

def generate_loss_and_constraints():
    H_generate = []
    y_generate = []
    A_generate = []
    b_generate = []

    for _ in range(total_steps):
        H = generate_H()
        one = np.ones((x_dim, 1))
        y = np.matmul(H, one) + np.random.normal(size=(4, 1))
        H_generate.append(H)
        y_generate.append(y)

        A = generate_A()
        b = generate_b()
        A_generate.append(A)
        b_generate.append(b)

    return H_generate, y_generate, A_generate, b_generate

def Sinha(step, choice, choice_aggregation, vq, device, gradientNorms):
    if step == 1:
        return np.zeros((x_dim, 1)), vq, 0

    H_estimate = H_list[device][step - 2]
    y_estimate = y_list[device][step - 2]
    A_estimate = A_list[device][step - 2]
    b_estimate = b_list[device][step - 2]

    V = 0.01
    lamda = 1 / (2 * np.sqrt(total_steps))
    G = 0.2
    D = 10 * np.sqrt(10)
    alpha = 1 / (2 * G * D)

    for i in range(constraint_dim):
        vq[i] = vq[i] + alpha * max(0, np.dot(A_estimate[i], choice[-1]) - b_estimate[i])

    ConsGradient = []
    for i in range(constraint_dim):
        if np.dot(A_estimate[i], choice[-1]) - b_estimate[i] > 0:
            ConsGradient.append(A_estimate[i])
        else:
            ConsGradient.append(np.zeros(x_dim))

    gradient =( V * alpha * np.dot(H_estimate.T, np.matmul(H_estimate, choice[-1]) - y_estimate).T 
            + (lamda * np.exp(lamda * vq[0]) * alpha * ConsGradient[0] + lamda * np.exp(lamda * vq[1]) * alpha * ConsGradient[1]) )

    gradient_norm = np.linalg.norm(gradient) ** 2
    gradientNorms = np.array(gradientNorms)
    eta = np.sqrt(2) * D / (2 * np.sqrt(gradientNorms.sum() + gradient_norm))

    x_new = choice_aggregation[-1] - eta * gradient.T
    for i in range(len(x_new)):
        if x_new[i] < -5:
            x_new[i] = -5
        if x_new[i] > 5:
            x_new[i] = 5
        else:
            x_new[i] = x_new[i]

    return x_new, vq, gradient_norm

# Sinha###Sinha#####Sinha####Sinha###Sinha#####Sinha######Sinha########Sinha######Sinha#####Sinha####Sinha#####Sinha######Sinha####
Avegraged_loss_Sinha = np.zeros(total_steps)
Avegraged_violation_Sinha = np.zeros(total_steps)
for total_run in range(run_number):
    H_list = [[] for n in range(N)]
    y_list = [[] for n in range(N)]
    A_list = [[] for n in range(N)]
    b_list = [[] for n in range(N)]
    for n in range(N):
        H_list[n], y_list[n], A_list[n], b_list[n] = generate_loss_and_constraints()

    WeightMatrix = []
    for step in range(total_steps):
        WeightMatrix.append(generate_WeightMatrix())

    choice_Sinha = [[] for _ in range(N)]
    choice_aggregation_Sinha = [[] for _ in range(N)]
    vq_Sinha = [np.zeros(constraint_dim) for _ in range(N)]
    x_value_Sinha = [np.zeros((x_dim, 1)) for _ in range(N)]
    list_of_gradientNorms = [[] for _ in range(N)]

    for step in range(1, total_steps + 1):
        for n in range(N):
            x_value_Sinha[n], vq_Sinha[n], gradientStore  = Sinha(step, choice_Sinha[n], choice_aggregation_Sinha[n], vq_Sinha[n], n, list_of_gradientNorms[n])
            list_of_gradientNorms[n].append(gradientStore)
        x_value_matrix = np.hstack(x_value_Sinha)
        aggregated_value = np.hsplit(np.dot(WeightMatrix[step - 1], x_value_matrix.T).T, N)
        for n in range(N):
            choice_aggregation_Sinha[n].append(aggregated_value[n])
            choice_Sinha[n].append(x_value_Sinha[n])

        if step % 200 == 0:
            print("Sinha: Run", total_run + 1, "Step", step, "finished.")

    accumulated_loss_Sinha = []
    total_loss_Sinha = 0
    for step in range(total_steps):
        for n in range(N):
            oneDevice_loss = 0
            x = choice_Sinha[n][step]
            for loss_n in range(N):
                H = H_list[loss_n][step]
                y = y_list[loss_n][step]
                loss = 0.5 * np.linalg.norm(np.matmul(H, x) - y)**2
                oneDevice_loss += loss/N
            total_loss_Sinha += oneDevice_loss/N
        accumulated_loss_Sinha.append(total_loss_Sinha)
    Avegraged_loss_Sinha = Avegraged_loss_Sinha + np.array(accumulated_loss_Sinha)

    accumulated_violation_Sinha = []
    total_violation_Sinha = 0
    for step in range(total_steps):
        for n in range(N):
            A = A_list[n][step]
            b = b_list[n][step]
            x = choice_Sinha[n][step]
            violation = np.linalg.norm(np.maximum(0, np.matmul(A, x) - b), 2) # L2 norm
            total_violation_Sinha += violation/N
        accumulated_violation_Sinha.append(total_violation_Sinha)
    Avegraged_violation_Sinha = Avegraged_violation_Sinha + np.array(accumulated_violation_Sinha)

Avegraged_loss_Sinha = Avegraged_loss_Sinha / (total_run + 1)
Avegraged_violation_Sinha = Avegraged_violation_Sinha / (total_run + 1)


# For saving and loading the results, uncomment the following lines as needed.

# with open('loss_timevarying_Sinha.pkl', 'wb') as f:
#     pickle.dump(Avegraged_loss_Sinha, f)

# with open('violation_timevarying_Sinha.pkl', 'wb') as f:
#     pickle.dump(Avegraged_violation_Sinha, f)

# with open('loss_timevarying_Sinha.pkl', 'rb') as f:
#     Avegraged_loss_Sinha = pickle.load(f)

# with open('violation_timevarying_Sinha.pkl', 'rb') as f:
#     Avegraged_violation_Sinha = pickle.load(f)



plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, total_steps + 1), Avegraged_loss_Sinha, label="[25]", color="orange", linestyle='-.')
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel("Accmulated Loss", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.grid(True)
plt.legend(prop={'family': 'Times New Roman', 'size': 20})
ax1 = plt.gca()
ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
for label in ax1.get_xticklabels():
    label.set_fontname('Times New Roman')
for label in ax1.get_yticklabels():
    label.set_fontname('Times New Roman')
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.yaxis.offsetText.set_fontsize(20)
ax1.yaxis.offsetText.set_fontname('Times New Roman')

plt.subplot(1, 2, 2)
plt.plot(range(1, total_steps + 1), Avegraged_violation_Sinha, label="[25]", color="orange", linestyle='-.')
plt.xlabel("Time", fontsize=24, fontdict={'fontname': 'Times New Roman'})
plt.ylabel("Hard Constraint Violation", fontsize=24, fontdict={'fontname': 'Times New Roman'})
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

plt.subplots_adjust(wspace=0.8)

plt.tight_layout()
plt.show()
