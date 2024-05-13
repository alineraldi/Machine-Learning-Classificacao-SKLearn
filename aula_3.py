# -*- coding: utf-8 -*-
"""Aula 3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16YVBctluVCuIQgJjI56ixdx3A59m3GSq
"""

import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/1b7d5475863c15f484ac495bd70975cf/raw/16aff7a0aee67e7c100a2a48b676a2d2d142f646/projects.csv"
dados=pd.read_csv(uri)
dados.head()

a_renomear = {
    "unfinished" : "nao_finalizado",
    "expected_hours" : "horas_esperadas",
    "price" : "preco"
}

dados = dados.rename(columns = a_renomear)
dados.head()
print(len(dados))

troca = {
    0: 1,
    1 : 0
}

dados["finalizado"] = dados.nao_finalizado.map(troca)
dados.tail()

import seaborn as sns

sns.scatterplot(x="horas_esperadas", y="preco", data=dados)

sns.scatterplot(x="horas_esperadas", y="preco", hue="finalizado", data=dados)

sns.relplot(x="horas_esperadas", y="preco", col="finalizado", hue="finalizado", data=dados)

x = dados [["horas_esperadas", "preco"]]
y = dados["finalizado"]

treino_x = x[:647]
treino_y = y[:647]

teste_x = x[647:]
teste_y = y[647:]

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

treino_x, treino_y, teste_x, teste_y, train_test_split (x, y,
                                                        random_state = SEED, test_size = 0.25,
                                                        stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A minha taxa de acertos foi de %.2f%%" % acuracia)

# teste para verificar se a acurácia da IA é tão boa quanto um chute apenas com 1 (linha de base)

import numpy as np
previsoes_de_base = np.ones(540)
acuracia = accuracy_score(teste_y, previsoes_de_base) * 100

print("A acurácio do algoritmo de baseline foi de %.2f%%" % acuracia)

sns.scatterplot(x="horas_esperadas", y="preco", hue=teste_y, data=teste_x)

x_min = teste_x.horas_esperadas.min()
x_max = teste_x.horas_esperadas.max()
y_min = teste_x.preco.min()
y_max = teste_x.preco.max()
print(x_min,x_max, y_min, y_max)

pixels = 100
eixo_x = np.arange(x_min, x_max, (x_max - x_min)/pixels)
eixo_y = np.arange(y_min, y_max, (y_max - y_min)/pixels)

xx, yy = np.meshgrid(eixo_x, eixo_y)
pontos = np.c_[yy.ravel(), xx.ravel()]
pontos

z = modelo.predict(pontos)
z = z.reshape(xx.shape)
z

import matplotlib.pyplot as plt

plt.contourf(xx, yy, z, alpha=0.3)
plt.scatter(teste_x.horas_esperadas, teste_x.preco, c=teste_y, s=1)

# DECISION BOUNDARY