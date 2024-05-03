# -*- coding: utf-8 -*-
"""Exercício 2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nc-ZQcUXV1sQ_1xLL2xueFvTdaIy7i6T
"""

import pandas as pd

uri = "https://raw.githubusercontent.com/alineraldi/Machine-Learning-Classificacao-SKLearn/main/exercicio2_mock_data.csv"
dados = pd.read_csv(uri)
dados.head()

# definindo features e labels

x = dados[["ave", "voa"]]
y = dados["herbivoro"]

dados.shape

#definindo treino e teste

treino_x = x[:70]
treino_y = y[:70]

teste_x = x[70:]
teste_y = y[70:]

print("Oi! Aqui é Octopus, a primeira IA programada por Aline Raldi \n Agora, demonstrarei minhas habilidades ao treinar %d elementos e testar %d elementos." % (len(treino_x), len(teste_x)))
print("Meu desafio é acertar quantas aves que voam são herbívoras. Vamos lá!")

# treinando a máquina

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100

print("A minha taxa de acertos foi de %.2f%%" % acuracia)

# train_test_split

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 20

treino_x, treino_y, teste_x, teste_y, train_test_split (x, y,
                                                        random_state = SEED, test_size = 0.25,
                                                        stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))