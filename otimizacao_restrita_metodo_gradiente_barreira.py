import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import numpy as np
from sympy import *
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
np.random.seed(0)

alpha = Symbol('a')

def gradiente_f(f, ponto):
    return nd.Gradient(f)(ponto)

def gradiente_numerico(f, ponto):
    delta = 0.0001
    n     = len(ponto)
    g     = [1] * n

    for i in range(n):
        aux_point = list(ponto)
        aux_point[i] += delta
        g[i] = round(((f(aux_point) - f(ponto))/delta), 5)

    return np.array(g)

def funcao_unidimensional(f, ponto):
    fl = f(ponto)
    return lambdify(alpha, fl, 'sympy')

def vetor_normalizado(vetor):
    N       = vetor.shape[0]
    pos     = np.array([0,0])
    dist    = np.zeros((N,1), dtype=np.float64)

    for i in range(N):
        norma = np.linalg.norm(vetor[i,:] - pos)
        dist[i] = norma

    return dist

def parada_vetor_de_variaveis(pontos_x):
    k = pontos_x.shape[0] - 1
    if(k  > 5):
        x_max           = max(pontos_x)
        x_min           = min(pontos_x)
        delta_x         = x_max - x_min
        vetor_ultimos_5 = [pontos_x[k], pontos_x[k-1], pontos_x[k-2], pontos_x[k-3], pontos_x[k-4], pontos_x[k-5]]
        x_max_5         = max(vetor_ultimos_5)
        x_min_5         = min(vetor_ultimos_5)
        gama_x          = x_max_5 - x_min_5

        return 1 if gama_x < 0.0001 * delta_x else 0
    else:
        return 0

def metodo_gradiente(func, ponto_inicial, mi = 100):
  i = mi;
  pos = 0;
  ponto = []

  while i >= 0:
    k = 0
    ponto.append([ponto_inicial])
    while 1:
        f = lambda x: func(i, x)
        if(parada_vetor_de_variaveis(vetor_normalizado(np.array(ponto[pos]))) == 1):
            print('Total de iteracoes: ', k - 1)
            break;

        gradiente              = -gradiente_f(f, ponto[pos][k])
        vetor_unidimensional   = ponto[pos][k] + np.dot(alpha, gradiente)
        func_unidimensional    = funcao_unidimensional(f, vetor_unidimensional)
        # Derivando em relação a alpha
        derivada_unidimensional = nd.Derivative(np.vectorize(lambdify(Symbol('x'), func_unidimensional(Symbol('x')), 'numpy')))
        valor_alpha             = min(fsolve(derivada_unidimensional, 0))

        proximo_ponto           = ponto[pos][k] + np.dot(valor_alpha, gradiente)
        print('mi:', i, 'x:', proximo_ponto[0], 'y:', proximo_ponto[1], 'f(x*):', f(proximo_ponto))
        ponto[pos].append(proximo_ponto)
        k  += 1
    pos += 1
    i -= 10

  return np.array(ponto)

def plot_metodo_gradiente(f, func_objetivo, func_penalidade, penalidade, ponto, mi):
    pontos_x = np.linspace(-30, 30, 100)

    pontos_metodo_gradiente = metodo_gradiente(f, ponto, mi)

    temp = mi;
    for j in range(len(pontos_metodo_gradiente)):
        fig = plt.figure()
        plt.title("Grafico dos valores do gradiente em função de mi")
        plt.plot(pontos_metodo_gradiente[j][:,0], pontos_metodo_gradiente[j][:,1], '-', label='mi: ' + str(temp))
        plt.xlabel('x1', fontsize=11)
        plt.ylabel('x2', fontsize=11)
        plt.legend(loc = "upper right")
        temp -= 10

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(pontos_x, pontos_x, func_objetivo(np.meshgrid(pontos_x, pontos_x)), 500, cmap='viridis')
    fig.suptitle('Grafico da função objetivo', fontsize=16)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(pontos_x, pontos_x, func_penalidade(np.meshgrid(pontos_x, pontos_x)), 500, cmap='viridis')
    fig.suptitle('Gráfico da função de restrição', fontsize=16)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(pontos_x, pontos_x, penalidade(mi, np.meshgrid(pontos_x, pontos_x)), 500, cmap='viridis')
    fig.suptitle('Gráfico da função de barreira', fontsize=16)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(pontos_x, pontos_x, f_restricted(mi, np.meshgrid(pontos_x, pontos_x)), 500, cmap='viridis')
    fig.suptitle('Gráfico da função restrita', fontsize=16)

    plt.show()

ponto = [2, 1]

def f(x):
    return (x[0]-1)**2 + 100*(x[1]-x[0]**2)**2
def f_penalidade(x):
    return (x[0]**2/(9)) + ((x[1]**2)/(16)) - 1

def penalidade(r, x):
    return r * ((-1)/f_penalidade(x))

def f_restricted(r, x):
    return f(x) + penalidade(r, x)

mi = 10
# metodo_gradiente(f_restricted, ponto, mi)
plot_metodo_gradiente(f_restricted, f, f_penalidade, penalidade, ponto, mi)
