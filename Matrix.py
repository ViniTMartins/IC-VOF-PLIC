import numpy as np
import math


def fun_limit_y(x,y):
    return funx(x) - (y - dy)
def funx(x):
    return func_ellipse(x,25,16,0,0)
def funy(y):
    return func_ellipse(y,16,25,0,0)

def func_ellipse(x,a,b,cx,cy):
    if abs(x) > a:
        return -100000
    root = 1 - ((x-cx) ** 2) / (a ** 2)
    y = cy + (b * math.sqrt(root))
    return y

def simpson (func, a, b, y):
    h = ((b - a)/6)
    x = h * (func(a,y) + (4 * func((a + b)/2, y)) + func(b,y))
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


def vof (x,y):
    v1x = x
    v2x = x + dx
    v3x = x
    v4x = x + dx
    v1y = y
    v2y = y
    v3y = y - dy
    v4y = y - dy

    bv1 = v1y < funx(v1x)
    bv2 = v2y < funx(v2x)
    bv3 = v3y < funx(v3x)
    bv4 = v4y < funx(v4x)
    sumB = bv1 + bv2 + bv3 + bv4
    if sumB == 0 :
        return 0
    if sumB == 4:
        return 1
    if sumB == 2:
        if bv1 == bv3:
            return (abs(simpson(fun_limit_y, funy(v1y), funy(v3y), y)) + (dy * (funy(v1y) - x)))/(dx * dy)
        else:
            return (abs(simpson(fun_limit_y, v1x, v2x, y)))/(dx * dy)
    if (sumB == 3) or (sumB == 1):

        if (bv1 != bv2) and (bv1 != bv3) and (bv1 != bv4):
            return (abs(simpson(fun_limit_y, v1x, funy(v1y), y)))/(dx * dy)
        if (bv2 != bv4) and (bv2 != bv1) and (bv2 != bv3):
            return (abs(simpson(fun_limit_y, v2x, funy(v2y), y)))/(dx * dy)
        if (bv3 != bv4) and (bv3 != bv1) and (bv3 != bv2):
            return (abs(simpson(fun_limit_y, v3x, funy(v3y), y)))/(dx * dy)
        else:
            return (abs(simpson(fun_limit_y, v4x, funy(v4y), y)))/(dx * dy)
    else:
        return -1







if __name__ == '__main__':
    dx = 2
    dy = 2
    n = 12
    arr = np.array([])
    linhas = []

    for i in range(n):
        for j in range(n):
            arr = np.append(arr, round(vof((dx * j), ((n * dy) - (dy * i))),1))
        linhas.append(arr)
        arr = np.array([])
    matrix = np.vstack(linhas)
    print(matrix)
    print(np.shape(matrix))

