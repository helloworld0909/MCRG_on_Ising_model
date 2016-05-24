import numpy as np
import math
import random

# 常数
J = 1
L = 128
BLOCK = 2
# fun_phi
NUM_SYSTEM = 128  # num of independent systems
STEP = 5  # sampling time divide
nRG = 4  # RG iteration times
# MCRG
NUM_ITERATION = 4  # 0 for Odd interaction


def Initial():
    global L
    Ising = np.empty((L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            Ising[i, j] = random.choice([-1, 1])
    return Ising


def Initialq():
    global L
    Ising = np.ones((L, L))
    if random.choice([-1, 1]) == -1:
        Ising *= -1
    return Ising


def Esingle(Ising, i, j):
    global J
    s = 0.0
    if i == Ising.shape[0] - 1:  # 边界周期性条件
        s += Ising[0, j]
    else:
        s += Ising[i + 1, j]
    if j == Ising.shape[1] - 1:
        s += Ising[i, 0]
    else:
        s += Ising[i, j + 1]
    s += (Ising[i - 1, j] + Ising[i, j - 1])  # 对于下标-1，python会自动处理
    return -Ising[i, j] * s * J


def E(Ising):
    global J
    energy = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            energy += Esingle(Ising, i, j)
    return 0.5 * energy  # 总能量是每个能量之和的一半


def Ssingle(Ising, i, j, alpha):  # 第alpha近的粒子对
    L1 = Ising.shape[0]
    L2 = Ising.shape[1]
    s = 0.0
    if alpha == 0:
        s = Ising[i, j]
    elif alpha == 1:
        if i == L1 - 1:  # 边界周期性条件
            s += Ising[0, j]
        else:
            s += Ising[i + 1, j]
        if j == L2 - 1:
            s += Ising[i, 0]
        else:
            s += Ising[i, j + 1]
        s += (Ising[i - 1, j] + Ising[i, j - 1])  # 对于下标-1，python会自动处理
        s *= (Ising[i, j] * 0.5)
    elif alpha == 2:
        x = i + 1
        y = j + 1
        if x == L1:
            x -= L1
        if y == L2:
            y -= L2
        s += (Ising[x, y] + Ising[x, j - 1] + Ising[i - 1, y] + Ising[i - 1, j - 1])
        s *= (Ising[i, j] * 0.5)
    elif alpha == 3:  # four spin
        s = Ising[i, j] * Ising[i - 1, j] * Ising[i, j - 1] * Ising[i - 1, j - 1]
    else:
        print('Error,alpha==' + str(alpha))
        exit()
    return s


def S_alpha(Ising, alpha):
    s = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            s += Ssingle(Ising, i, j, alpha)
    return s


def BlockIsing(Ising, b):
    if Ising.shape[0] % b != 0 or Ising.shape[1] % b != 0:
        return "error b"
    IsingB = np.empty((Ising.shape[0] // b, Ising.shape[1] // b))
    for i in range(Ising.shape[0] // b):
        for j in range(Ising.shape[1] // b):
            if sum(sum(Ising[b * i:b * i + b, b * j:b * j + b])) > 0:
                IsingB[i, j] = 1
            elif sum(sum(Ising[b * i:b * i + b, b * j:b * j + b])) < 0:
                IsingB[i, j] = -1
            else:
                IsingB[i, j] = random.choice([-1, 1])
    return IsingB


def Pflip(Ising, i, j, T):
    dH = -2 * Esingle(Ising, i, j)
    return min(math.exp(-dH / T), 1)


def MCpass(Ising, T):  # 一个pass随机取L*L次粒子
    global L
    for n in range(L * L):
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        P = Pflip(Ising, i, j, T)
        if P == 1:
            Ising[i, j] *= -1
        else:
            r = random.random()
            if r < P:
                Ising[i, j] *= -1
    return 0


def Ebtw(Ising, Ising0, i, j):
    if Ising.shape[0] != Ising0.shape[0]:
        print('error')
    global J
    s = 0.0
    if i == Ising.shape[0] - 1:  # 边界周期性条件
        s += Ising0[0, j]
    else:
        s += Ising0[i + 1, j]
    if j == Ising.shape[1] - 1:
        s += Ising0[i, 0]
    else:
        s += Ising0[i, j + 1]
    s += (Ising0[i - 1, j] + Ising0[i, j - 1])  # 对于下标-1，python会自动处理
    return Ising[i, j] * s * J


def funE(Ising, Ising0):
    N = Ising.shape[0] * Ising.shape[1]
    sum = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            sum += Ebtw(Ising, Ising0, i, j)
    return sum / N


def mag(Ising):
    M = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            M += Ising[i][j]
    return M


def magavg(IsingG):
    global NUM_SYSTEM
    avg = 0.0
    for j in range(NUM_SYSTEM):
        avg += mag(IsingG[j])
    avg = avg / NUM_SYSTEM
    return avg


def funPhi(IsingG, IsingG0):
    global NUM_SYSTEM
    M0avg = magavg(IsingG0)
    Mtavg = magavg(IsingG)

    M0Mtavg = 0.0
    M0sigma = 0.0
    Mtsigma = 0.0
    for j in range(NUM_SYSTEM):
        M0 = mag(IsingG0[j])
        Mt = mag(IsingG[j])
        M0Mtavg += M0 * Mt
        M0sigma += (M0 - M0avg) ** 2
        Mtsigma += (Mt - Mtavg) ** 2

    M0Mtavg = M0Mtavg / NUM_SYSTEM
    M0sigma = math.sqrt(M0sigma / NUM_SYSTEM)
    Mtsigma = math.sqrt(Mtsigma / NUM_SYSTEM)

    phi = (M0Mtavg - M0avg * Mtavg) / M0sigma / Mtsigma
    return phi


def matrixAB(IsingGroup):
    global NUM_ITERATION, BLOCK
    n = len(IsingGroup)
    matrixA = np.empty((NUM_ITERATION, NUM_ITERATION))
    matrixB = np.empty((NUM_ITERATION, NUM_ITERATION))
    S = np.zeros((NUM_ITERATION, n))
    SR = np.zeros((NUM_ITERATION, n))

    for j in range(n):
        Tmp = BlockIsing(IsingGroup[j], BLOCK)
        for a in range(NUM_ITERATION):
            S[a, j] = S_alpha(IsingGroup[j], a)
            SR[a, j] = S_alpha(Tmp, a)
    for a in range(NUM_ITERATION):
        for b in range(NUM_ITERATION):
            matrixA[a, b] = sum(S[b] * SR[a]) / n - sum(S[b]) * sum(SR[a]) / (n * n)
            matrixB[a, b] = sum(SR[b] * SR[a]) / n - sum(SR[b]) * sum(SR[a]) / (n * n)
    return matrixA, matrixB


def write_nparray(file, array):
    for j in range(array.shape[1]):
        for i in range(array.shape[0]):
            file.write(str(array[i, j]) + "\t")
        file.write("\n")
    return 0
