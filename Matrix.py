import numpy as np
import math


def fun(x):
    return x*x

def simpson (func, a, b):
    h = ((b - a)/6)
    x = h * (func(a) + (4 * func((a + b)/2)) + func(b))
    return x



def clasificacao (arr, iElem, jElem):
    maxI, maxJ = arr.shape
    iStart = max(0, iElem - 1)
    iEnd = min(maxI, jElem + 2)
    jStart = max(0, jElem - 1)
    jEnd = min(maxJ, jElem + 2)
    corte = arr[iStart:iEnd, jStart:jEnd]

    if arr[iElem][jElem] == 0 :
        for i in np.nditer(corte):
            if i != 0 :
                return 'I'
        return 'V'
    if  arr[iElem][jElem] == 1 :
        for i in np.nditer(corte):
            if i != 0:
                return 'I'
        return 'V'
    else:
        return 'I'


def vof (x):
    if(x == 1):
        return 1
    if(x == 0):
        return 0
    else:
        return x


if __name__ == '__main__':
    n = int(input())
    arr = np.array([])
    for i in range(n):
        arr = np.append(arr, vof(float(0)))
    matrix = np.reshape(arr, (math.ceil(math.sqrt(n)), math.ceil(math.sqrt(n))))

    print(matrix)
    print(np.shape(matrix))


    print(simpson( fun, 0, 1))
