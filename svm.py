#!/usr/bin/env python3

import numpy as np

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
def carregar_base(nome_arquivo):
    x = []
    y = []

    with open(nome_arquivo, "r") as arquivo:
        for linha in arquivo:
            valores = linha.split()

            classe = int(valores[0])
            y.append(classe)

            atributos = []

            for valor in valores[1:]:
                indice, numero = valor.split(":")
                atributos.append(float(numero))

            x.append(atributos)
    return np.array(x, dtype=np.float32), np.array(y)

print("Carregando conjunto de treinamento...")
x_train, y_train = carregar_base("train.txt")

print("Carregando conjunto de teste...")
x_test, y_test = carregar_base("test.txt")

print("Treinando SVM")


# Quantidade de exemplos utilizada em cada treinamento
passo = 1000

# Lista para guardar os tamanhos das bases
tamanhos_treinamento = []

# Lista para guardar as acurácias
acuracias = []


print("\nTestando diferentes tamanhos da base de treinamento...\n")

print("Treinando Arvore de decisão")

tamanho_total = len(x_train)

for tamanho in range(passo, tamanho_total + 1, passo):

    print(f"Treinando com {tamanho} exemplos...")

    # Seleciona somente os primeiros 'tamanho' exemplos
    x_train_atual = x_train[:tamanho]
    y_train_atual = y_train[:tamanho]

    # Cria um novo classificador
    modelo = svm.SVC()

    # Treina o modelo
    modelo.fit(x_train_atual, y_train_atual)

    # Classifica o conjunto de teste
    y_pred = modelo.predict(x_test)

    # Calcula a acurácia
    acuracia = accuracy_score(y_test, y_pred)

    # Guarda os resultados
    tamanhos_treinamento.append(tamanho)
    acuracias.append(acuracia)

    print(f"Acurácia: {acuracia:.4f}\n")

    print("\nEvaluation:")
    print(classification_report(y_test, y_pred))
    # print(classification_report(y_validation, predictions))
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred)
    # cm = confusion_matrix(y_validation, predictions)
    print (cm)

plt.figure(figsize=(10, 6))

plt.plot(
    tamanhos_treinamento,
    acuracias,
    marker="o"
)

plt.xlabel("Tamanho da base de treinamento")
plt.ylabel("Acurácia")
plt.title(
    f"Desempenho do SVM em função do tamanho da base "
)

plt.ylim(0, 1.05)
plt.grid(True)

plt.tight_layout()
plt.show()


