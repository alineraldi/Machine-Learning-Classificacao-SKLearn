# -*- coding: utf-8 -*-

#importando os dados
import pandas as pd

uri = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dados = pd.read_csv(uri)
dados.head()

#renomeando as colunas
mapa = {
    "home" : "principal",
    "how_it_works" : "como_funciona",
    "contact" : "contato",
    "bought" : "comprou"
}
dados = dados.rename(columns = mapa)

# definindo features
# seleciona apenas as colunas descritas, precisa estar numa array quando for mais de 1 coluna
x = dados[["principal","como_funciona", "contato"]]
x.head()

# definindo a label
y = dados["comprou"]
y.head()

dados.shape # verifica a quantidade de elementos

# definindo quais os dados de treino e de teste

treino_x = x[:75]
treino_y = y[:75]

teste_x = x[75:]
teste_y = y[75:]

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

#  treinando a máquina
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)

# testando a máquina

previsoes = modelo.predict(teste_x)

# definindo a taxa de acerto

acuracia = accuracy_score(teste_y, previsoes)
print("A acurácia foi %.2f%%" % acuracia)

"""# Usando a biblioteca para separar treino e teste"""

# ao invés daquelas 4 linhas de código, vamos separar o treino do teste de uma forma mais fácil
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

treino_x, teste_x, treino_y, teste_y, train_test_split(x, y, random_state = SEED, test_size = 0.25)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# a função train_test_split usam separações aleatórias, os resultados podem ser diferentes

treino_y.value_counts()

teste_y.value_counts()

# ao invés daquelas 4 linhas de código, vamos separar o treino do teste de uma forma mais fácil
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

SEED = 20

treino_x, teste_x, treino_y, teste_y, train_test_split(x, y,
                                                        random_state = SEED, test_size = 0.25,
                                                        stratify = y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

modelo = LinearSVC()
modelo.fit(treino_x, treino_y)
previsoes = modelo.predict(teste_x)

acuracia = accuracy_score(teste_y, previsoes) * 100
print("A acurácia foi %.2f%%" % acuracia)

# a função train_test_split usam separações aleatórias, os resultados podem ser diferentes

treino_y.value_counts()

teste_y.value_counts()