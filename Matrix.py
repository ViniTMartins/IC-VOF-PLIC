import numpy as np
import math


def fun_limit_y(x,y):
    return funx(x,y) - y
def funx(x,y):
    if y > 0:
        return func_ellipse_top(x, 16, 16, 0, 0)
    else:
        return func_ellipse_bottom(x, 16, 16, 0, 0)

def funy(y,x):
    if x > 0:
        return func_ellipse_inverse_right(y, 16, 16, 0, 0)
    else:
        return func_ellipse_inverse_left(y, 16, 16, 0, 0)

def func_ellipse_top(x,a,b,cx,cy):
    if abs(x - cx) > a:
        return -100000
    root = 1 - ((x-cx) ** 2) / (a ** 2)
    y = cy + (b * math.sqrt(root))
    return y
def func_ellipse_bottom(x,a,b,cx,cy):
    if abs(x - cx) > a:
        return 100000
    root = 1 - ((x-cx) ** 2) / (a ** 2)
    y = cy - (b * math.sqrt(root))
    return y
def func_ellipse_inverse_left(y,a,b,cx,cy):
    if abs(y - cy) > b:
        return 100000
    root = 1 - ((y-cy) ** 2) / (b ** 2)
    x = cx - (a * math.sqrt(root))
    return x
def func_ellipse_inverse_right(y,a,b,cx,cy):
    if abs(y - cy) > b:
        return -100000
    root = 1 - ((y-cy) ** 2) / (b ** 2)
    x = cx + (a * math.sqrt(root))
    return x

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
    #
    #     v1-----------------v2
    #     |                   |
    # dy  |                   |
    #     |                   |
    #     |                   |
    #     v3-----------------v4
    #               dx
    v1x = x - dx/2
    v2x = x + dx/2
    v3x = x - dx/2
    v4x = x + dx/2
    v1y = y + dy/2
    v2y = y + dy/2
    v3y = y - dy/2
    v4y = y - dy/2
    bottom_y = y - dy/2
    if v1y <= 0:
        v1_in = v1y > funx(v1x, v1y)
    else:
        v1_in = v1y < funx(v1x, v1y)
    if v2y <= 0:
        v2_in = v2y > funx(v2x,v2y)
    else:
        v2_in = v2y < funx(v2x, v2y)
    if v3y <= 0:
        v3_in = v3y > funx(v3x,v3y)
    else:
        v3_in = v3y < funx(v3x, v3y)
    if v4y <= 0:
        v4_in = v4y > funx(v4x,v4y)
    else:
        v4_in = v4y < funx(v4x, v4y)

    sumB = v1_in + v2_in + v3_in + v4_in

    if sumB == 0 :
        return 0
    elif sumB == 4:
        return 1
    elif sumB == 2:
        if v1_in == v3_in == 1:
            return (abs(simpson(fun_limit_y, funy(v1y, v1x), funy(v3y, v3x), bottom_y)) + (dy * (min(funy(v1y, v1x),funy(v3y,v3x)) - v1x)))/(dx * dy)
        elif v1_in == v3_in == 0:
            return 1 - ((abs(simpson(fun_limit_y, funy(v1y, v1x), funy(v3y, v3x), bottom_y)) + (dy * (min(funy(v1y, v1x),funy(v3y,v3x)) - v1x)))/(dx * dy))
        elif v4_in == v3_in == 1:
            return (abs(simpson(fun_limit_y, v1x, v2x, bottom_y)))/(dx * dy)
        else:
            return 1 - (abs(simpson(fun_limit_y, v1x, v2x, bottom_y))) / (dx * dy)

    elif sumB == 1:

        if (v1_in != v2_in) and (v1_in != v3_in) and (v1_in != v4_in):
            return 1 - (((abs(simpson(fun_limit_y, v1x, funy(v1y, v1x), bottom_y))) + (dy * (dx - (funy(v1y, v1x) - x)))) /(dx * dy))
        elif (v2_in != v4_in) and (v2_in != v1_in) and (v2_in != v3_in):
            return 1 - (((abs(simpson(fun_limit_y, v2x, funy(v2y, v2y), bottom_y))) + (dy * (funy(v1y, v1x) - x))) /(dx * dy))
        elif (v3_in != v4_in) and (v3_in != v1_in) and (v3_in != v2_in):
            return (abs(simpson(fun_limit_y, v3x, funy(v3y, v3x), bottom_y)))/(dx * dy)
        else:
            return (abs(simpson(fun_limit_y, v4x, funy(v4y, v4x), bottom_y)))/(dx * dy)
    elif sumB == 3:
        if (v1_in != v2_in) and (v1_in != v3_in) and (v1_in != v4_in):
            return ((abs(simpson(fun_limit_y, v1x, funy(v1y, v1x), bottom_y))) + (dy * (dx - (funy(v1y, v1x) - x)))) /(dx * dy)
        elif (v2_in != v4_in) and (v2_in != v1_in) and (v2_in != v3_in):
            return ((abs(simpson(fun_limit_y, v2x, funy(v2y, v2x), bottom_y))) + (dy * (funy(v1y, v1x) - x))) /(dx * dy)
        elif (v3_in != v4_in) and (v3_in != v1_in) and (v3_in != v2_in):
            return 1 - ((abs(simpson(fun_limit_y, v3x, funy(v3y, v3x), bottom_y))) / (dx * dy))
        else:
            return 1 - ((abs(simpson(fun_limit_y, v4x, funy(v4y, v4x), bottom_y))) / (dx * dy))
    else:
        return -1







if __name__ == '__main__':

    #X and Y values for each cell
    #     |-------------------|
    #     |                   |
    # dy  |                   |
    #     |                   |
    #     |                   |
    #     |-------------------|
    #               dx
    dx = 2
    dy = 2
    n_x = 12
    n_y = 12
    off_center_x = 0
    off_center_y = -10
    matrix = np.zeros((n_y, n_x))

    for i in range(n_y):
        for j in range(n_x):
            x_cell = ((dx * j) + dx/2) + off_center_x
            y_cell = (n_y * dy) - (dy * i) - dy/2 + off_center_y
            matrix[i,j] = round(vof(x_cell,y_cell),1)

    print(matrix)
    print(np.shape(matrix))

