import random
import numpy as np
import cvxpy as cp
import math
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import ScalarFormatter
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# Seed Setting
set_seed(44)

x_dim = 2
constraint_dim = 3
A_range = (0, 0.5)
b_range = (0, 0.3)
total_steps = 1000
N = 20  # number of devices
rho = 0.1  # connection probability

run_number = 20

x_upper = 1
x_lower = -1

ut = random.sample(range(1, total_steps + 1), total_steps)

def generate_Theta1(t):
    return np.random.uniform(low=-math.pow(t, 1 / 10), high=math.pow(t, 1 / 10), size=(x_dim, 1))

def generate_Theta2(t):
    if t in range(1, 301):
        return np.random.uniform(low=-1, high=0, size=(x_dim, 1))
    if t in range(400, 701):
        return np.random.uniform(low=-1, high=0, size=(x_dim, 1))
    if t in range(800, 1000):
        return np.random.uniform(low=-1, high=0, size=(x_dim, 1))
    else:
        return np.random.uniform(low=0, high=1, size=(x_dim, 1))

def generate_Theta3(t):
    Theta3 = np.zeros((x_dim,1))
    Theta3[0] = math.pow(-1, ut[t - 1])
    Theta3[1] = math.pow(-1, ut[t - 1])
    return Theta3

def generate_A():
    return np.random.uniform(low=A_range[0], high=A_range[1], size=(constraint_dim, x_dim))

def generate_b():
    return np.random.uniform(low=b_range[0], high=b_range[1], size=(constraint_dim, 1))

def generate_WeightMatrix():
    P = np.zeros((N, N))
    for i in range(N - 1):
        P[i + 1, i] = 1 / N

    for i in range(N):
        for j in range(N):
            if i != j and np.random.rand() <= rho:
                P[i, j] = 1 / N
        P[i, i] = 1 - np.sum(P[i, :])
    return P

def generate_loss_and_constraints():
    Theta_generate = []
    A_generate = []
    b_generate = []

    for slot in range(1, total_steps + 1):
        Theta1 = generate_Theta1(slot)
        Theta2= generate_Theta2(slot)
        Theta3 = generate_Theta3(slot)
        Theta_sum = Theta1 +Theta2 + Theta3
        Theta_generate.append(Theta_sum)

    A = generate_A()
    b = generate_b()
    A_generate.append(A)
    b_generate.append(b)

    return Theta_generate, A_generate, b_generate

def RECOO(step, choice, choice_aggregation, vq, device):
    if step == 1:
        return np.zeros((x_dim, 1)), vq

    Theta_estimate = Theta_list[device][step - 2]
    A_estimate = A_list[device][0]
    b_estimate = b_list[device][0]

    alpha = 2 * np.sqrt(step - 1)
    eta = 2 * np.sqrt(step)
    gamma = 0.5 * math.pow(step - 1, 1 / 2 + 0.01)

    lossGradient = (2 * (choice[-1] - Theta_estimate) + 20 * Theta_estimate).T

    x = cp.Variable(shape=(x_dim, 1))

    constraints = [x >= x_lower, x <= x_upper]
    objective = cp.Minimize(
        lossGradient @ (x - choice_aggregation[-1])
        + gamma * sum(
            vq[i] * (cp.abs(A_estimate[i] @ x - b_estimate[i]) + A_estimate[i] @ x - b_estimate[i]) / 2
            for i in range(constraint_dim)
        )
        + alpha * cp.square(cp.norm(x - choice_aggregation[-1]))
    )

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    x_new = x.value

    A_estimate_current = A_list[device][0]
    b_estimate_current = b_list[device][0]
    gamma_current = 0.5 * math.pow(step, 1 / 2 + 0.01)

    for i in range(constraint_dim):
        vq[i] = max(
            eta,
            vq[i] + gamma_current * max(0, np.dot(A_estimate_current[i], x_new) - b_estimate_current[i])
        )

    return x_new, vq

# ##RECOO##############RECOO#############RECOO############RECOO##############RECOO#########RECOO#####
Avegraged_loss_RECOO = np.zeros(total_steps)
Avegraged_violation_RECOO = np.zeros(total_steps)
for total_run in range(run_number):
    start_time = time.time()
    Theta_list = [[] for n in range(N)]
    A_list = [[] for n in range(N)]
    b_list = [[] for n in range(N)]

    for n in range(N):
        Theta_list[n], A_list[n], b_list[n] = generate_loss_and_constraints()

    WeightMatrix = []
    for step in range(total_steps):
        WeightMatrix.append(generate_WeightMatrix())

    choice_RECOO = [[] for _ in range(N)]
    choice_aggregation_RECOO = [[] for _ in range(N)]
    vq_RECOO = [np.zeros(constraint_dim) for _ in range(N)]
    x_value_RECOO = [np.zeros((x_dim, 1)) for _ in range(N)]

    for step in range(1, total_steps + 1):
        for n in range(N):
            x_value_RECOO[n], vq_RECOO[n] = RECOO(step, choice_RECOO[n], choice_aggregation_RECOO[n], vq_RECOO[n], n)
        x_value_matrix = np.hstack(x_value_RECOO)
        aggregated_value = np.hsplit(np.dot(WeightMatrix[step - 1], x_value_matrix.T).T, N)
        for n in range(N):
            choice_aggregation_RECOO[n].append(aggregated_value[n])
            choice_RECOO[n].append(x_value_RECOO[n])

        if step % 200 == 0:
            print("RECOO: Run", total_run + 1, "Step", step, "finished.")

    accumulated_loss_RECOO = []
    total_loss_RECOO = 0
    for step in range(total_steps):
        for n in range(N):
            oneDevice_loss = 0
            x = choice_RECOO[n][step]
            for loss_n in range(N):
                Theta = Theta_list[loss_n][step]
                loss = (x - Theta).T @ (x - Theta) + 20 * Theta.T @ x
            total_loss_RECOO += loss[0,0] / N
        accumulated_loss_RECOO.append(total_loss_RECOO)
    Avegraged_loss_RECOO = Avegraged_loss_RECOO + np.array(accumulated_loss_RECOO)

    accumulated_violation_RECOO = []
    total_violation_RECOO = 0
    for step in range(total_steps):
        for n in range(N):
            A = A_list[n][0]
            b = b_list[n][0]
            x = choice_RECOO[n][step]
            violation = np.linalg.norm(np.maximum(0, np.matmul(A, x) - b), 2)
            total_violation_RECOO += violation / N
        accumulated_violation_RECOO.append(total_violation_RECOO)
    Avegraged_violation_RECOO = Avegraged_violation_RECOO + np.array(accumulated_violation_RECOO)

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time:.2f} seconds.")

Avegraged_loss_RECOO = Avegraged_loss_RECOO / (total_run + 1)
Avegraged_violation_RECOO = Avegraged_violation_RECOO / (total_run + 1)


# For saving and loading the results, uncomment the following lines as needed.

# with open('loss_fixedCons_RECOO.pkl', 'wb') as f:
#     pickle.dump(Avegraged_loss_RECOO, f)

# with open('violation_fixedCons_RECOO_.pkl', 'wb') as f:
#     pickle.dump(Avegraged_violation_RECOO, f)

# with open('loss_fixedCons_RECOO.pkl', 'rb') as f:
#     Avegraged_loss_RECOO = pickle.load(f)

# with open('violation_fixedCons_RECOO_.pkl', 'rb') as f:
#     Avegraged_violation_RECOO = pickle.load(f)


plt.figure(figsize=(13, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, total_steps + 1), Avegraged_loss_RECOO, label="[21]", color="blue", linestyle='--')
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
plt.plot(range(1, total_steps + 1), Avegraged_violation_RECOO, label="[24]", color="blue", linestyle='--')
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
