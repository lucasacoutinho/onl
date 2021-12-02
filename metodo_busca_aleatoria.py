import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')

import numpy as np
from sympy import *
import numdifftools as nd
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
np.random.seed(0)

alpha = Symbol('a')

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

        return 1 if gama_x < 0.0000000000000000000001 * delta_x else 0
    else:
        return 0

def busca_aleatoria(f, ponto_inicial, n = 2):
  k               = 0
  ponto           = []
  ponto.append(np.array(ponto_inicial))

  while 1:
    if(parada_vetor_de_variaveis(vetor_normalizado(np.array(ponto))) == 1):
      print('Total de iteracoes: ', k - 1)
      return np.array(ponto)

    direcao_random           = np.random.normal(0, 1, n)
    vetor_unidimensional     = ponto[k] + np.dot(alpha, direcao_random)
    func_unidimensional      = funcao_unidimensional(f, vetor_unidimensional)
    # Com derivada da função unidimensional
    derivada_unidimensional  = nd.Derivative(np.vectorize(lambdify(var('x'), func_unidimensional(var('x')), 'numpy')))
    valor_alpha              = min(fsolve(derivada_unidimensional, 0))

    proximo_ponto   = ponto[k] + valor_alpha * direcao_random
    print('x', proximo_ponto[0], 'y', proximo_ponto[1], 'f(x*)', f(proximo_ponto))
    ponto.append(np.array(proximo_ponto))
    k += 1

  return np.array(ponto)

def plot(f, numpy_function, ponto, n = 2):
    pontos_x = np.linspace(-5.12, 5.12, 100)

    pontos_busca_aleatoria = busca_aleatoria(f, ponto, n)
    pontos_busca_aleatoria_x = [a_tuple[0] for a_tuple in pontos_busca_aleatoria]
    pontos_busca_aleatoria_y = [a_tuple[1] for a_tuple in pontos_busca_aleatoria]
    pontos_busca_aleatoria_z = []
    for j in range(len(pontos_busca_aleatoria)):
        pontos_busca_aleatoria_z.append(f(pontos_busca_aleatoria[j]))

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Metodo de busca aleatória")
    plt.contour(pontos_x, pontos_x, numpy_function(np.meshgrid(pontos_x, pontos_x)), colors='black')
    plt.plot(pontos_busca_aleatoria_x[0], pontos_busca_aleatoria_y[0], 'ro', label='ponto inicial')
    plt.plot(pontos_busca_aleatoria_x, pontos_busca_aleatoria_y, linewidth=2.0, label='busca aleatória')
    plt.plot(pontos_busca_aleatoria_x[-1], pontos_busca_aleatoria_y[-1], 'bo' ,label='ponto final')
    plt.xlabel('x1', fontsize=11)
    plt.ylabel('x2', fontsize=11)
    plt.legend(loc = "upper right")

    plt.subplot(1, 2, 2)
    plt.plot(range(len(pontos_busca_aleatoria)), pontos_busca_aleatoria_z)
    plt.title("Grafico de convergência")

    plt.show()

##### OBS:
## Para fazer a plotagem é necessário converter as expressões do sympy para numpy
# Ou seja, Abs -> np.abs
# sin  -> np.sin
# pi   -> np.pi
# exp  -> np.exp
# sqrt -> np.sqrt

def f_CROSS_IN_TRAY(x):
    a=Abs(100-sqrt(x[0]*x[0]+x[1]*x[1])/pi)
    b=Abs(sin(x[0]) * sin(x[1])*exp(a))+1
    c=-0.0001*b**0.1
    return c

def numpy_function_CROSS_IN_TRAY(x):
    a=np.fabs(100-np.sqrt(x[0]*x[0]+x[1]*x[1])/np.pi)
    b=np.fabs(np.sin(x[0]) * np.sin(x[1])*np.exp(a))+1
    c=-0.0001*b**0.1
    return c

def f_ROSENBROCK(x):
    return (x[0]-1)**2 + 100*(x[1]-x[0]**2)**2

def numpy_function_ROSENBROCK(x):
    return (x[0]-1)**2 + 100*(x[1]-x[0]**2)**2

def f_BEALE(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

def numpy_function_BEALE(x):
    return (1.5 - x[0] + x[0]*x[1])**2 + (2.25 - x[0] + x[0]*x[1]**2)**2 + (2.625 - x[0] + x[0]*x[1]**3)**2

ponto = [-5, 1]
## PLOT FUNCTION CROSS_IN_TRAY
# plot(f_CROSS_IN_TRAY, numpy_function_CROSS_IN_TRAY, ponto)
## PLOT FUNCTION ROSENBROCK
plot(f_ROSENBROCK, numpy_function_ROSENBROCK, ponto)
## PLOT FUNCTION BEALE
# plot(f_BEALE, numpy_function_BEALE, ponto)
