import random
import numpy as np
import matplotlib.pyplot as plt

GENE_NUM = 50
GENERATION = 5000
Y_MAX = 1.0e5
THRESHOLD = 1.0e-5
MIN = -2.048
MAX = 2.048

def noDuplicationRandInt(min, max, num):
    indexList = list(range(min, max + 1))
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
    ax.set_title("optimal solution serch by RGA")
    ax.legend()
    plt.savefig(fileName + ".png")

class Indivisual:
    def __init__(self, geneNum):
        self.gene = np.random.rand(GENE_NUM) * (MAX - MIN) + MIN

    def rosenBrock(self):
        return np.sum(100 * (self.gene[0] - self.gene[1:] ** 2) ** 2 + (1 - self.gene[1:]) ** 2)

class RGA:
    def __init__(self, k, indivisualNum):
        self.k = k
        self.indivisualNum = indivisualNum
        self.indivisuals = [Indivisual(GENE_NUM) for i in range(self.indivisualNum)]
        self.parentIndex = None
        self.parentSize = GENE_NUM + self.k

    def execRGA(self):
        generationList = []
        bestValueList = []
        loop = 0
        while loop < GENERATION and self.calcFitness() > THRESHOLD:
            if loop % 100 == 0:
                print("generation : %d  min : %f" % (loop, self.calcFitness()))
                print(sorted(self.indivisuals, key = lambda x: x.rosenBrock())[0].gene)
                generationList.append(loop)
                bestValueList.append(self.calcFitness())
            self.execJGG()
            #self.generateMutation()
            loop += 1
        print("generation : %d  min : %f" % (loop, self.calcFitness()))
        print(sorted(self.indivisuals, key = lambda x: x.rosenBrock())[0].gene)
        generationList.append(loop)
        bestValueList.append(self.calcFitness())
        return (generationList, bestValueList)


    def calcFitness(self):
        return min(self.indivisuals, key = lambda x: x.rosenBrock()).rosenBrock()

    def execJGG(self):
        self.parentIndex = noDuplicationRandInt(0, self.indivisualNum - 1, self.parentSize)
        parent = [self.indivisuals[i] for i in self.parentIndex]
        child = self.execREX(parent, GENE_NUM * 10)
        eliteChild = self.selectElite(child, self.parentSize)
        self.exchangeIndividual(eliteChild)

    def execREX(self, parent, childNum):
        child = []
        parentGene = list(map(lambda x: x.gene, parent))
        centerOfGravity = np.sum(parentGene, axis = 0) / len(parentGene)
        while len(child) < childNum:
            tmp = Indivisual(GENE_NUM)
            tmp.gene = centerOfGravity + np.sum(np.random.normal(0, np.sqrt(1.0 / self.parentSize), (len(parentGene), 1)) * (parentGene - centerOfGravity), axis = 0)
            while min(tmp.gene) < MIN:
                tmp.gene[tmp.gene.argmin()] = MIN
            while max(tmp.gene) > MAX:
                tmp.gene[tmp.gene.argmax()] = MAX
            child.append(tmp)
        return child

    def selectElite(self, indivisual, num):
        elite = sorted(indivisual, key = lambda x:x.rosenBrock())
        return elite[:num]

    def exchangeIndividual(self, child):
        self.indivisuals = [self.indivisuals[i] for i in range(self.indivisualNum) if i not in self.parentIndex]
        self.indivisuals.extend(child)

    def execRGAAndSaveResut(self, ax, label):
        print(label)
        generationList, bestValueList = self.execRGA()
        ax.plot(generationList, bestValueList, label = label)

if __name__ == "__main__":
    kList = [1, 10, 50, 100, 500]
    indList = [100, 500, 1000, 5000, 10000]
    fig, ax = plt.subplots()
    for k in kList:
        ga = RGA(k, indList[2])
        ga.execRGAAndSaveResut(ax, "k=" + str(k))
    saveResult(ax, "ga_k")
    fig, ax = plt.subplots()
    for i in indList:
        ga = RGA(kList[2], i)
        ga.execRGAAndSaveResut(ax, "indivisual=" + str(i))
    saveResult(ax, "ga_indivisual")
