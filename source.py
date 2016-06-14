import numpy as np
import math
import random

# free parameters in common
J = 1
L = 64
BLOCK = 2

# fun_phi
NUM_SYSTEM = 128  # num of independent systems
STEP = 5  # sampling time divide
nRG = 4  # RG iteration times

# MCRG
NUM_INTERACTION = 8  # 0 for Odd interaction


def Initial():
    global L
    Ising = np.empty((L, L, L), dtype=int)
    for i in range(L):
        for j in range(L):
            for k in range(L):
                Ising[i, j, k] = random.choice([-1, 1])
    return Ising


def Esingle(Ising, i, j, k):
    global J
    s = 0.0
    x = i + 1
    y = j + 1
    z = k + 1
    if x == Ising.shape[0]:  # 边界周期性条件
        x -= Ising.shape[0]
    if y == Ising.shape[1]:
        y -= Ising.shape[1]
    if z == Ising.shape[2]:
        z -= Ising.shape[2]
    s += (Ising[i - 1, j, k] + Ising[i, j - 1, k] + Ising[i, j, k - 1] + Ising[x, j, k] + Ising[i, y, k] + Ising[
        i, j, z])  # 对于下标-1，python会自动处理
    return -Ising[i, j, k] * s * J


def E(Ising):
    global J
    energy = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            for k in range(Ising.shape[2]):
                energy += Esingle(Ising, i, j, k)
    return 0.5 * energy  # 总能量是每个能量之和的一半


def Ssingle(Ising, i, j, k, alpha):  # 第alpha近的粒子对
    L1 = Ising.shape[0]
    L2 = Ising.shape[1]
    L3 = Ising.shape[2]
    s = 0.0
    if alpha == 0:
        s = Ising[i, j, k]
    elif alpha == 1:
        x = i + 1
        y = j + 1
        z = k + 1
        if x == L1:  # 边界周期性条件
            x -= L1
        if y == L2:
            y -= L2
        if z == L3:
            z -= L3
        s += (Ising[i - 1, j, k] + Ising[i, j - 1, k] + Ising[i, j, k - 1] + Ising[x, j, k] + Ising[i, y, k] + Ising[
            i, j, z])  # 对于下标-1，python会自动处理
        s *= (Ising[i, j, k] * 0.5)
    elif alpha == 2:
        x = i + 1
        y = j + 1
        z = k + 1
        if x == L1:
            x -= L1
        if y == L2:
            y -= L2
        if z == L3:
            z -= L3
        s += (
            Ising[x, y, k] + Ising[x, j - 1, k] + Ising[i - 1, y, k] + Ising[i - 1, j - 1, k] + Ising[x, j, z] + Ising[
                x, j, k - 1] + Ising[i - 1, j, z] + Ising[i - 1, j, k - 1] + Ising[i, y, z] + Ising[i, j - 1, z] +
            Ising[i, y, k - 1] + Ising[i, j - 1, k - 1])
        s *= (Ising[i, j, k] * 0.5)
    elif alpha == 3:
        x = i + 1
        y = j + 1
        z = k + 1
        if x == L1:
            x -= L1
        if y == L2:
            y -= L2
        if z == L3:
            z -= L3
        s += (
            Ising[x, y, z] + Ising[i - 1, y, z] + Ising[x, j - 1, z] + Ising[x, y, k - 1] + Ising[i - 1, j - 1, z] +
            Ising[i - 1, y, k - 1] + Ising[x, j - 1, k - 1] + Ising[i - 1, j - 1, k - 1])
        s *= (Ising[i, j, k] * 0.5)
    elif alpha == 4:  # four spins nearest neighbor
        s += Ising[i, j - 1, k] * Ising[i, j, k - 1] * Ising[i, j - 1, k - 1] + Ising[i - 1, j, k] * Ising[
            i, j, k - 1] * Ising[i - 1, j, k - 1] + Ising[i - 1, j, k] * Ising[i - 1, j, k] * Ising[i - 1, j - 1, k]
        s *= Ising[i, j, k]
    elif alpha == 5:  # four spins next-nearest neighbor
        x = i + 1
        y = j + 1
        z = k + 1
        if x == L1:
            x -= L1
        if y == L2:
            y -= L2
        if z == L3:
            z -= L3
        s += Ising[i, j - 1, k - 1] * Ising[i, j - 2, k] * Ising[i, j - 1, z] + Ising[i - 1, j, k - 1] * Ising[
            i, j, k - 2] * Ising[x, j, k - 1] + Ising[i - 1, j - 1, k] * Ising[i - 2, j, k] * Ising[i - 1, y, k]
        s *= Ising[i, j, k]
    elif alpha == 6:  # four spins tetrahedral vertices in each cube
        s += Ising[i - 1, j, k] * Ising[i, j - 1, k] * Ising[i, j, k - 1] * Ising[i - 1, j - 1, k - 1] + \
             Ising[i - 1, j - 1, k] * Ising[i, j - 1, k - 1] * Ising[i - 1, j, k - 1] * Ising[i, j, k]
    elif alpha == 7:
        s += Ising[i - 2, j, k] + Ising[i, j - 2, k] + Ising[i, j, k - 2]
        s *= Ising[i, j, k]
    else:
        print('Error,alpha==' + str(alpha))
        exit()
    return s


def S_alpha(Ising, alpha):
    s = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            for k in range(Ising.shape[2]):
                s += Ssingle(Ising, i, j, k, alpha)
    return s


def BlockIsing(Ising, b):
    if Ising.shape[0] % b != 0 or Ising.shape[1] % b != 0 or Ising.shape[2] % b != 0:
        return "error b"
    IsingB = np.empty((Ising.shape[0] // b, Ising.shape[1] // b, Ising.shape[2] // b))
    for i in range(Ising.shape[0] // b):
        for j in range(Ising.shape[1] // b):
            for k in range(Ising.shape[2] // b):
                s = sum(sum(sum(Ising[b * i:b * i + b, b * j:b * j + b, b * k:b * k + b])))
                if s > 0:
                    IsingB[i, j, k] = 1
                elif s < 0:
                    IsingB[i, j, k] = -1
                else:
                    IsingB[i, j, k] = random.choice([-1, 1])
    return IsingB


def Pflip(Ising, i, j, k, T):
    dH = -2 * Esingle(Ising, i, j, k)
    return min(math.exp(-dH / T), 1)


def MCpass(Ising, T):  # 一个pass随机取L*L次粒子
    global L
    for n in range(L * L * L):
        i = random.randint(0, L - 1)
        j = random.randint(0, L - 1)
        k = random.randint(0, L - 1)
        P = Pflip(Ising, i, j, k, T)
        if P == 1:
            Ising[i, j, k] *= -1
        else:
            r = random.random()
            if r < P:
                Ising[i, j, k] *= -1
    return 0


'''unfinished
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
'''


def mag(Ising):
    M = 0.0
    for i in range(Ising.shape[0]):
        for j in range(Ising.shape[1]):
            for k in range(Ising.shape[2]):
                M += Ising[i][j][k]
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
    global NUM_INTERACTION, BLOCK
    n = len(IsingGroup)
    matrixA = np.empty((NUM_INTERACTION, NUM_INTERACTION))
    matrixB = np.empty((NUM_INTERACTION, NUM_INTERACTION))
    S = np.zeros((NUM_INTERACTION, n))
    SR = np.zeros((NUM_INTERACTION, n))

    for j in range(n):
        Tmp = BlockIsing(IsingGroup[j], BLOCK)
        for a in range(NUM_INTERACTION):
            S[a, j] = S_alpha(IsingGroup[j], a)
            SR[a, j] = S_alpha(Tmp, a)
    for a in range(NUM_INTERACTION):
        for b in range(NUM_INTERACTION):
            matrixA[a, b] = sum(S[b] * SR[a]) / n - sum(S[b]) * sum(SR[a]) / (n * n)
            matrixB[a, b] = sum(SR[b] * SR[a]) / n - sum(SR[b]) * sum(SR[a]) / (n * n)
    return matrixA, matrixB


def write_nparray(file, array):
    for j in range(array.shape[1]):
        for i in range(array.shape[0]):
            file.write(str(array[i, j]) + "\t")
        file.write("\n")
    return 0
