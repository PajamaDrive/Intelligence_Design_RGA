import numpy as np
import numba

def comp():
    pass

@numba.njit
def noDuplicationRandIntWithoutCertainValue(min, max, num, value):
    indexList = list(range(min, max + 1))
    indexList.remove(value)
    result = np.zeros(num)
    for i in range(num):
        result[i] = indexList.pop(np.random.randint(0, len(indexList) - 1))
    


if __name__ == "__main__":
    print(noDuplicationRandIntWithoutCertainValue(0, 10, 9, 2))