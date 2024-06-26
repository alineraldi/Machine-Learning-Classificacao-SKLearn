# -*- coding: utf-8 -*-
"""Exercício 1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1IcQY0umkrR0BTj4AB_buMiW2RHfseFUg
"""

# sim 1, nao 0
# professora inglês = 1 - professor música = 0

# gosta de estudar?
# domina um instrumento?
# estuda gramática?

professormusica1 = [1,1,0]
professormusica2 = [0,1,0]
professormusica3 = [1,1,1]

professoraingles1 = [0,1,1]
professoraingles2 = [1,0,1]
professoraingles3 = [0,0,1]

#treino

treino_x = [professormusica1, professormusica2, professormusica3, professoraingles1, professoraingles2, professoraingles3]
treino_y = [0,0,0,1,1,1]

from sklearn.svm import LinearSVC

model = LinearSVC()
model.fit(treino_x,treino_y)

# previsão

professormisterioso = [1,1,0]
model.predict([professormisterioso])

misterio1 = [0,1,1]
misterio2 = [1,1,0]
misterio3 = [1,1,1]

teste_x = [misterio1, misterio2, misterio3]
teste_y = [0,0,1]

previsoes = model.predict(teste_x)
print(model.predict(teste_x))

corretos = (previsoes == teste_y).sum()
total = len(teste_x)
print("Taxa de acertos: %.2f" % ((corretos/total) * 100), "%")

from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acertos: %.2f" % (taxa_de_acerto * 100), "%")