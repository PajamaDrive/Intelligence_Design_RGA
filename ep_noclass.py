import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import numba

GENE_NUM = 50
GENERATION = 5000
Y_MAX = 1.0e5
THRESHOLD = 1.0e-5
MIN = -2.048
MAX = 2.048

def execEPAndSaveResut(m, indivusalNum, sigma, ax, label):
    print(label)
    generationList, bestValueList = execEP(m, indivusalNum, sigma)
    ax.plot(generationList, bestValueList, label = label)

def execEP(m, indivisualNum, sigma):
    indivisuals = np.array([[random.uniform(MIN, MAX) for i in range(GENE_NUM)] for i in range(indivisualNum)])
    fitness = calcFitness(indivisuals)
    loop = 0
    generationList = []
    bestValueList = []
    while loop < GENERATION and fitness.min() > THRESHOLD:
        if loop % 100 == 0:
            print("generation : %d  min : %f" % (loop, fitness.min()))
            print(indivisuals[fitness.argmin()])
            generationList.append(loop)
            bestValueList.append(fitness.min())
        indivisuals = np.append(indivisuals, generateMutation(copy.deepcopy(indivisuals), indivisualNum, sigma), axis = 0)
        fitness = calcFitness(indivisuals)
        survivalValue = calcSurvivalValue(fitness, 2 * indivisualNum, m)
        indivisuals = selectEliteIndivisual(indivisuals, survivalValue, indivisualNum)
        fitness = calcFitness(indivisuals)
        loop += 1
    print("generation : %d  min : %f" % (loop, fitness.min()))
    print(indivisuals[fitness.argmin()])
    generationList.append(loop)
    bestValueList.append(fitness.min())
    return (generationList, bestValueList)

def calcFitness(indivisuals):
    return np.array(list(map(rosenBrock, indivisuals)))

def rosenBrock(gene):
    return np.sum(100 * (gene[0] - gene[1:] ** 2) ** 2 + (1 - gene[1:]) ** 2)

@numba.njit
def generateMutation(indivisuals, indivisualNum, sigma):
    for i in range(indivisualNum):
        indivisuals[i] += np.random.normal(0, sigma, GENE_NUM)
        if indivisuals[i].min() < MIN or indivisuals[i].max() > MAX: 
            for j in range(GENE_NUM):
                if indivisuals[i][j] < MIN:
                    indivisuals[i][j] = MIN
                if indivisuals[i][j] > MAX:
                    indivisuals[i][j] = MAX
    return indivisuals

@numba.njit
def calcSurvivalValue(fitness, indivisualNum, m):
    survivalValue = np.zeros(indivisualNum)
    for i in range(indivisualNum):
        indexList = list(range(0, indivisualNum))
        indexList.remove(i)
        selectedIndivisualIndex = np.zeros(m)
        for j in range(m):
            selectedIndivisualIndex[j] = indexList.pop(np.random.randint(0, len(indexList) - 1))
        for x in selectedIndivisualIndex:
            if fitness[i] <= fitness[int(x)]:
                survivalValue[i] += 1
    return survivalValue

def selectEliteIndivisual(indivisuals, survivalValue, indivisualNum):
    eliteIndivisuals = np.zeros((indivisualNum, GENE_NUM))
    for i in range(indivisualNum):
        eliteIndivisuals[i] = indivisuals[survivalValue.argmax()]
        indivisuals = np.delete(indivisuals, survivalValue.argmax(), axis = 0)
        survivalValue = np.delete(survivalValue, survivalValue.argmax())
    return eliteIndivisuals


def saveResult(ax, fileName):
    ax.set_yscale("log")
    plt.ylim(THRESHOLD, Y_MAX)
    plt.xlim(0, GENERATION)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title("optimal solution serch by EP")
    ax.legend()
    plt.savefig(fileName + ".png")

if __name__ == "__main__":
    mList = [10, 50, 100, 500]
    indList = [100, 500, 1000, 5000]
    sigmaList = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1]
    fig, ax = plt.subplots()
    for m in mList:
        execEPAndSaveResut(m, indList[1], sigmaList[3], ax, "m=" + str(m))
    saveResult(ax, "ep_noclass_adjust_m")
    fig, ax = plt.subplots()
    for i in indList:
        execEPAndSaveResut(mList[2], i, sigmaList[3], ax, "indivisual=" + str(i))
    saveResult(ax, "ep_noclass_adjust_indivisual")
    """
    fig, ax = plt.subplots()
    for s in sigmaList:
        execEPAndSaveResut(mList[2], indList[2], s, ax, "sigma=" + str(s))
    saveResult(ax, "ep_noclass_sigma")
    """