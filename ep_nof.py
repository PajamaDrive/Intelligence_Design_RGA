import random
import numpy as np
import copy
import matplotlib.pyplot as plt

GENE_NUM = 50
GENERATION = 5000
Y_MAX = 1.0e5
THRESHOLD = 1.0e-5
MIN = -2.048
MAX = 2.048

def noDuplicationRandIntWithoutCertainValue(min, max, num, value):
    indexList = list(range(min, max + 1))
    indexList.remove(value)
    result = []
    while len(result) < num:
        result.append(indexList.pop(random.randint(0, len(indexList) - 1)))
    return result

def saveResult(ax, fileName):
    ax.set_yscale("log")
    plt.ylim(THRESHOLD, Y_MAX)
    plt.xlim(0, GENERATION)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Value")
    ax.set_title("optimal solution serch by EP")
    ax.legend()
    plt.savefig(fileName + ".png")

class Indivisual:
    def __init__(self, geneNum):
        self.gene = np.array([random.uniform(MIN, MAX) for i in range(geneNum)])
        self.survivalValue = None

    def rosenBrock(self):
        return np.sum([100 * (self.gene[0] - self.gene[i] ** 2) ** 2 + (1 - self.gene[i]) ** 2 for i in range(1, len(self.gene))])

class EP:
    def __init__(self, m, indivisualNum, sigma):
        self.m = m
        self.indivisualNum = indivisualNum
        self.sigma = sigma
        self.indivisuals = [Indivisual(GENE_NUM) for i in range(self.indivisualNum)]

    def execEP(self):
        loop = 0
        generationList = []
        bestValueList = []
        while loop < GENERATION and self.calcFitness() > THRESHOLD:
            if loop % 100 == 0:
                print("generation : %d  min : %f" % (loop, self.calcFitness()))
                print(sorted(self.indivisuals, key = lambda x: x.rosenBrock())[0].gene)
                generationList.append(loop)
                bestValueList.append(self.calcFitness())
            copyIndivisual = copy.deepcopy(self.indivisuals)
            self.indivisuals.extend(list(map(self.generateMutation, copyIndivisual)))
            self.calcSurvivalValue()
            self.indivisuals.sort(key = lambda x: x.survivalValue, reverse = True)
            del self.indivisuals[self.indivisualNum:]
            loop += 1
        print("generation : %d  min : %f" % (loop, self.calcFitness()))
        print(sorted(self.indivisuals, key = lambda x: x.rosenBrock())[0].gene)
        generationList.append(loop)
        bestValueList.append(self.calcFitness())
        return (generationList, bestValueList)

    def calcFitness(self):
        return min(self.indivisuals, key = lambda x: x.rosenBrock()).rosenBrock()

    def generateMutation(self, indivisual):
        indivisual.gene += np.random.normal(0, np.sqrt(self.sigma), GENE_NUM)
        if min(indivisual.gene) < MIN  or max(indivisual.gene) > MAX:
            indivisual.gene = list(map(lambda x: MAX if x > MAX else (MIN if x < MIN else x), indivisual.gene))
        return indivisual

    def calcSurvivalValue(self):
        for i in range(len(self.indivisuals)):
            selectedIndivisualIndex = noDuplicationRandIntWithoutCertainValue(0, len(self.indivisuals) - 1, self.m, i)
            self.indivisuals[i].survivalValue = sum(self.indivisuals[i].rosenBrock() <= self.indivisuals[x].rosenBrock() for x in selectedIndivisualIndex)

    def execEPAndSaveResut(self, ax, label):
        print(label)
        generationList, bestValueList = self.execEP()
        ax.plot(generationList, bestValueList, label = label)

if __name__ == "__main__":
    mList = [5, 10, 50, 100]
    indList = [100, 500, 1000, 5000]
    fig, ax = plt.subplots()
    for m in mList:
        ep = EP(m, indList[1], 0.01)
        ep.execEPAndSaveResut(ax, "m=" + str(m))
    saveResult(ax, "ep_m")
    fig, ax = plt.subplots()
    for i in indList:
        ep = EP(mList[1], i, 0.01)
        ep.execEPAndSaveResut(ax, "indivisual=" + str(i))
    saveResult(ax, "ep_indivisual")
    