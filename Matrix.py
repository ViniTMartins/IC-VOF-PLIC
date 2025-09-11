import numpy as np
from sympy import symbols, diff
import math


def get_params():
    #return [a, b, cx, cy]
    params = [5,5,0,0]
    return params

def fun_limit_y(x,bottom_y, y_cell,dx,dy):
    return funx(x,y_cell, dx,dy) - bottom_y
def funx(x,y_cell,dx,dy):
    arr = get_params()
    return func_ellipse(x, arr[0], arr[1], arr[2], arr[3], y_cell, dx,dy)

def funy(y,x_cell, dx,dy):
    arr = get_params()
    return func_ellipse(y, arr[1], arr[0], arr[3], arr[2], x_cell, dx,dy)
def check_in(x,y, dx,dy):
    arr = get_params()
    return inside_ellipse(x,y, arr[0], arr[1], arr[2], arr[3], dx,dy)

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
    #     v 3-----------------v4 <- bottom_y
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


def classificacao (matrix, method):

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

def normal(x_cell, y_cell,i,j, classificacao):
    x, y = symbols('x, y', real=True)
    params = get_params()
    f = (x - params[2])**2/params[0]**2 + (y - params[3])**2/params[1]**2 - 1
    pdfx = diff(f, x)
    pdfy = diff(f, y)
    if classificacao[i, j] == 'I':
        return pdfx.subs({x: x_cell, y: y_cell}), pdfy.subs({x: x_cell, y: y_cell})
    else:
        return 0,0




def salvar_vtk_celula(
    caminho_arquivo: str,
    campo_escalar: np.ndarray,
    campo_vetorial: np.ndarray,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    cx: float,
    cy: float,
    nome_campo: str = "campo_escalar",
    nome_vetorial: str = "Normals"
):
    """
    Salva um campo escalar 2D associado às células de uma malha estruturada
    regular em um arquivo de formato VTK (legacy ASCII).

    Args:
        caminho_arquivo (str): O caminho completo para o arquivo de saída (ex: 'dados.vtk').
        campo_escalar (np.ndarray): Matriz 2D com os valores do campo escalar.
                                    A dimensão deve ser (ny, nx).
        nx (int): Número de pontos na direção x.
        ny (int): Número de pontos na direção y.
        dx (float): Espaçamento entre os pontos na direção x.
        dy (float): Espaçamento entre os pontos na direção y.
        cx (float): Coordenada x da origem da malha (canto inferior esquerdo).
        cy (float): Coordenada y da origem da malha (canto inferior esquerdo).
        nome_campo (str, optional): Nome do campo escalar a ser salvo no arquivo VTK.
                                    Padrão é "campo_escalar".
    """
    # Validação das dimensões do campo escalar
    num_celulas_y, num_celulas_x = campo_escalar.shape
    if num_celulas_x != nx or num_celulas_y != ny :
        raise ValueError(
            f"A dimensão do campo escalar ({num_celulas_y}, {num_celulas_x}) "
            f"não é compatível com o número de células ({ny}, {nx})."
        )

    num_celulas = nx * ny

    try:
        with open(caminho_arquivo, 'w') as f:
            # --- 1. Cabeçalho VTK ---
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"Malha 2D Estruturada - {nome_campo}\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")

            # --- 2. Informação Geométrica/Topológica ---
            # Dimensões (número de PONTOS)
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            # Origem da malha (coordenada do primeiro ponto)
            f.write(f"ORIGIN {cx} {cy} 0.0\n")
            # Espaçamento entre os pontos
            f.write(f"SPACING {dx} {dy} 1.0\n")

            # --- 3. Dados das Células ---
            f.write(f"\nPOINT_DATA {num_celulas}\n")
            # Definição do campo escalar
            f.write(f"SCALARS {nome_campo} float 1\n")
            f.write("LOOKUP_TABLE default\n")

            # --- 4. Escrita dos dados ---
            # O VTK espera que os dados sejam escritos com o índice x variando mais
            # rapidamente. O método flatten() do numpy com a ordem 'C' (padrão)
            # faz exatamente isso.
            campo_achatado = campo_escalar.flatten(order='C')
            for valor in campo_achatado:
                f.write(f"{valor}\n")

            """f.write(f"\nVECTORS {nome_vetorial} float\n")
            campo_vec_flat = campo_vetorial.reshape(-1, 2)  # (nx*ny, 2)
            for vx, vy in campo_vec_flat:
                f.write(f"{vx} {vy} 0.0\n")"""

        print(f"Arquivo '{caminho_arquivo}' salvo com sucesso!")

    except IOError as e:
        print(f"Erro ao escrever o arquivo '{caminho_arquivo}': {e}")






def inicio():
    # X and Y values for each cell
    #     |-------------------|
    #     |                   |
    # dy  |                   |
    #     |                   |
    #     |                   |
    #     |-------------------|
    #               dx
    dx = 0.5
    dy = 0.5
    n_x = 100
    n_y = 100
    off_center_x = -4
    off_center_y = -10
    matrix = np.zeros((n_y, n_x))
    normals = np.empty((n_y, n_x), dtype='object')


    for i in range(n_y):
        for j in range(n_x):
            x_center_of_cell = ((dx * j) + dx / 2) + off_center_x
            y_center_of_cell = ((dy * i) + dy / 2) + off_center_y
            matrix[i, j] = vof(x_center_of_cell, y_center_of_cell, dx,dy)


    print(matrix)
    print(np.shape(matrix))

    classificacao_matrix = classificacao(matrix, 'matrix')

    for i in range(n_y):
        for j in range(n_x):
            x_center_of_cell = ((dx * j) + dx / 2) + off_center_x
            y_center_of_cell = ((dy * i) + dy / 2) + off_center_y
            normals[i, j] = normal(x_center_of_cell, y_center_of_cell, i, j, classificacao_matrix)

    print(classificacao_matrix)
    print(normals)

    salvar_vtk_celula("output.vtk", matrix, normals, n_x, n_y, dx, dy, off_center_x, off_center_y, "Volume_Fraction", "Normals"
    )


if __name__ == '__main__':
    inicio()


