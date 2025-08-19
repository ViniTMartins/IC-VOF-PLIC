import numpy as np
import math


def fun_limit_y(x,bottom_y, y_cell,dx,dy):
    return funx(x,y_cell, dx,dy) - bottom_y
def funx(x,y_cell,dx,dy):
    return func_ellipse(x, 16, 16, 0, 0, y_cell, dx,dy)

def funy(y,x_cell, dx,dy):
    return func_ellipse(y, 16, 16, 0, 0, x_cell, dx,dy)
def check_in(x,y, dx,dy):
    return inside_ellipse(x,y, 16,16,0,0, dx,dy)

def func_ellipse(x,a,b,cx,cy,y_cell,dx,dy):
    if abs(x - cx) > a:
        return float("inf")
    root = 1 - ((x-cx) ** 2) / (a ** 2)
    y = cy + (b * math.sqrt(root))
    if abs(y - y_cell) > dy:
        return y - 2 * (y - cy)
    return y

def inside_ellipse(x,y,a,b,cx,cy, dx,dy):
    part_x = (((x-cx)/a) ** 2)
    part_y = (((y-cy)/b) ** 2)
    if part_x + part_y <= 1 + 1e-12:
        return True
    else:
        return False

"""
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
"""

def simpson (func, a, b, bottom_y,y_cell, dx,dy):
    h = ((b - a)/6)
    x = h * (func(a,bottom_y,y_cell, dx,dy) + (4 * func((a + b)/2, bottom_y,y_cell, dx,dy)) + func(b,bottom_y,y_cell, dx,dy))
    return x

def calc_area (function, ini_x, final_x, orientation, inner, bottom_y, y_cell, area_rectangle, dx,dy):
    #     ---------------------
    #     |################### | <- orientation = upper
    #     |###############     |
    #     |##########          |
    #     |#####               |
    #     |##                  |
    #     ---------------------
    #
    #     ---------------------
    #     |         ##########| <- orientation = lower
    #     |       ############|
    #     |      #############|
    #     |   ################|
    #     |###################|
    #      --------------------
    if orientation == 'L':
        result = abs(simpson(function, ini_x, final_x, bottom_y, y_cell, dx,dy))
    elif orientation == 'U':
        result = abs(simpson(function, ini_x, final_x, (bottom_y + dy), y_cell, dx,dy))
    else:
        result = 0

    result_norm = abs((result + area_rectangle) / (dx * dy))
    if inner:
        return result_norm
    else:
        return 1 - result_norm

'''
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
'''


def vof (x,y, dx,dy):
    #
    #     v1-----------------v2 <- upper_x
    #     |                   |
    #     |                   |
    #  dy |       (x,y)       |
    #     |                   |
    #     |                   |
    #     v3-----------------v4 <- bottom_y
    #     ^         dx        ^
    #     |                   |
    #     left_x              right_x
    #
    # v1 = (upper_y,left_x)
    # v2 = (upper_y,right_x)
    # v3 = (lower_y,left_x)
    # v4 = (lower_y,right_x)
    #

    lower_y = y - dy/2
    upper_y = y + dy/2
    left_x = x - dx/2
    right_x = x + dx/2

    v1_in = check_in(left_x, upper_y, dx,dy)
    v2_in = check_in(right_x, upper_y, dx,dy)
    v3_in = check_in(left_x, lower_y, dx,dy)
    v4_in = check_in(right_x, lower_y, dx,dy)

    sumB = v1_in + v2_in + v3_in + v4_in

    area_rectangle = 0

    if sumB == 0 :
        #     ---------------------
        #     |                    |
        #     |                    |
        #     |                    |
        #     |                    |
        #     |                    |
        #      ---------------------
        return 0
    elif sumB == 4:
        #     ---------------------
        #     |####################|
        #     |####################|
        #     |####################|
        #     |####################|
        #     |####################|
        #      ---------------------
        return 1
    elif sumB == 2:
        if v1_in == v3_in == 1:
            #     -----------|---------
            #     |##########|         |
            #     |##########|         |
            #     |##########|         |
            #     |##########|         |
            #     |##########|         |
            #      ----------|----------

            #return (abs(simpson(fun_limit_y, funy(v1y, v1x), funy(v3y, v3x), bottom_y)) + (dy * (min(funy(v1y, v1x),funy(v3y,v3x)) - v1x)))/(dx * dy)
            ini_x = funy(upper_y, x, dx,dy)
            final_x = funy(lower_y, x, dx,dy)
            area_rectangle = (min(ini_x, final_x) - left_x) * dy
            if min(ini_x, final_x) == final_x:
                orientation = 'L'
            else:
                orientation = 'U'
            inner = True

        elif v1_in == v3_in == 0:
            #     -----------|---------
            #     |          |#########|
            #     |          |#########|
            #     |          |#########|
            #     |          |#########|
            #     |          |#########|
            #      ----------|----------

            # return 1 - ((abs(simpson(fun_limit_y, funy(v1y, v1x), funy(v3y, v3x), bottom_y)) + (dy * (min(funy(v1y, v1x),funy(v3y,v3x)) - v1x)))/(dx * dy))
            ini_x = funy(upper_y, x, dx,dy)
            final_x = funy(lower_y, x, dx,dy)
            area_rectangle = (right_x - max(ini_x, final_x)) * dy
            if min(ini_x, final_x) == final_x:
                orientation = 'U'
            else:
                orientation = 'L'
            inner = True
        elif v4_in == v3_in == 1:
            #     ---------------------
            #     |                    |
            #     |                    |
            #     ----------------------
            #     |####################|
            #     |####################|
            #      ---------------------

            # return (abs(simpson(fun_limit_y, v1x, v2x, bottom_y)))/(dx * dy)
            ini_x = left_x
            final_x = right_x
            orientation = 'L'
            # unecessary for this case
            inner = True
        else:
            #     ---------------------
            #     |####################|
            #     |####################|
            #     ----------------------
            #     |                    |
            #     |                    |
            #      ---------------------
            # return 1 - (abs(simpson(fun_limit_y, v1x, v2x, bottom_y))) / (dx * dy)
            ini_x = left_x
            final_x = right_x
            orientation = 'U'
            # unecessary for this case
            inner = True

    elif sumB == 1:

        if (v1_in != v2_in) and (v1_in != v3_in) and (v1_in != v4_in):
            #     ----------|----------
            #     |#########/          |
            #     ----------           |
            #     |                    |
            #     |                    |
            #     |                    |
            #      ---------------------
            # return 1 - (((abs(simpson(fun_limit_y, v1x, funy(v1y, v1x), bottom_y))) + (dy * (dx - (funy(v1y, v1x) - x)))) /(dx * dy))
            ini_x = left_x
            final_x = funy(upper_y, x, dx,dy)
            orientation = 'U'
            inner = True

        elif (v2_in != v4_in) and (v2_in != v1_in) and (v2_in != v3_in):
            #     ----------|-----------
            #     |         \##########|
            #     |          -----------
            #     |                    |
            #     |                    |
            #     |                    |
            #      ---------------------
            # return 1 - (((abs(simpson(fun_limit_y, v2x, funy(v2y, v2y), bottom_y))) + (dy * (funy(v1y, v1x) - x))) /(dx * dy))
            ini_x = funy(upper_y, x, dx,dy)
            final_x = right_x
            orientation = 'U'
            inner = True

        elif (v3_in != v4_in) and (v3_in != v1_in) and (v3_in != v2_in):
            #     ---------------------
            #     |                    |
            #     |                    |
            #     |                    |
            #     -----------          |
            #     |##########\         |
            #      ----------|---------
            # return (abs(simpson(fun_limit_y, v3x, funy(v3y, v3x), bottom_y)))/(dx * dy)
            ini_x = left_x
            final_x = funy(lower_y, x, dx,dy)
            orientation = 'L'
            inner = True

        else:
            #     ---------------------
            #     |                    |
            #     |                    |
            #     |                    |
            #     |           ----------
            #     |          /#########|
            #      ----------|----------
            # return (abs(simpson(fun_limit_y, v4x, funy(v4y, v4x), bottom_y)))/(dx * dy)
            ini_x = funy(lower_y, x, dx,dy)
            final_x = right_x
            orientation = 'L'
            inner = True
    elif sumB == 3:

        if (v1_in != v2_in) and (v1_in != v3_in) and (v1_in != v4_in):
            #     ----------|----------
            #     |         /##########|
            #     ----------###########|
            #     |####################|
            #     |####################|
            #     |####################|
            #      ---------------------
            # return ((abs(simpson(fun_limit_y, v1x, funy(v1y, v1x), bottom_y))) + (dy * (dx - (funy(v1y, v1x) - x)))) /(dx * dy)
            ini_x = left_x
            final_x = funy(upper_y, x, dx,dy)
            orientation = 'U'
            inner = False


        elif (v2_in != v4_in) and (v2_in != v1_in) and (v2_in != v3_in):
            #     ----------|-----------
            #     |#########\          |
            #     |##########-----------
            #     |####################|
            #     |####################|
            #     |####################|
            #      ---------------------
            # return ((abs(simpson(fun_limit_y, v2x, funy(v2y, v2x), bottom_y))) + (dy * (funy(v1y, v1x) - x))) /(dx * dy)
            ini_x = funy(upper_y, x, dx,dy)
            final_x = right_x
            orientation = 'U'
            inner = False

        elif (v3_in != v4_in) and (v3_in != v1_in) and (v3_in != v2_in):
            #     ---------------------
            #     |####################|
            #     |####################|
            #     |####################|
            #     -----------##########|
            #     |          \#########|
            #      ----------|---------
            # return 1 - ((abs(simpson(fun_limit_y, v3x, funy(v3y, v3x), bottom_y))) / (dx * dy))
            ini_x = left_x
            final_x = funy(lower_y, x, dx,dy)
            orientation = 'L'
            inner = False

        else:
            #     ---------------------
            #     |####################|
            #     |####################|
            #     |####################|
            #     |###########----------
            #     |##########/         |
            #      ----------|----------
            #  return 1 - ((abs(simpson(fun_limit_y, v4x, funy(v4y, v4x), bottom_y))) / (dx * dy))
            ini_x = funy(lower_y, x, dx,dy)
            final_x = right_x
            orientation = 'L'
            inner = False
    else:
        return -1

    return calc_area(fun_limit_y, ini_x, final_x, orientation, inner, lower_y, y, area_rectangle, dx,dy)


def classificacao (matrix):

    n_y, n_x = matrix.shape

    matrix_classificacao = np.empty((n_y, n_x), dtype='<U1')

    for i in range(n_y):
        for j in range(n_x):
            valor = matrix[i][j]

            if 0 < valor < 1:
                matrix_classificacao[i][j] = 'I'

            elif valor == 1:

                if i > 0 and matrix[i-1][j] == 0:
                    matrix_classificacao[i][j] = 'I'

                elif i < n_y - 1 and matrix[i+1][j] == 0:
                    matrix_classificacao[i][j] = 'I'

                elif j > 0 and matrix[i][j-1] == 0:
                    matrix_classificacao[i][j] = 'I'

                elif j < n_x - 1 and matrix[i][j+1] == 0:
                    matrix_classificacao[i][j] = 'I'

                else:
                    matrix_classificacao[i][j] = 'F'

            else:
                matrix_classificacao[i][j] = 'V'

    return matrix_classificacao





def inicio():
    # X and Y values for each cell
    #     |-------------------|
    #     |                   |
    # dy  |                   |
    #     |                   |
    #     |                   |
    #     |-------------------|
    #               dx
    dx = 2
    dy = 2
    n_x = 10
    n_y = 10
    off_center_x = -16
    off_center_y = 0
    matrix = np.zeros((n_y, n_x))

    for i in range(n_y):
        for j in range(n_x):
            x_center_of_cell = ((dx * j) + dx / 2) + off_center_x
            y_center_of_cell = ((dy * i) + dy / 2) + off_center_y
            matrix[i, j] = round(vof(x_center_of_cell, y_center_of_cell, dx,dy), 1)

    print(matrix)
    print(np.shape(matrix))

    classificacao_matrix = classificacao(matrix)

    print(classificacao_matrix)




if __name__ == '__main__':
    inicio()


