#-*- coding: utf-8 -*-
##------------------------------------------------------------------------
## Case: Cobrança - Comparação de Técnicas
## Autor: Prof. Roberto Angelo
## Objetivo: Cross-validation e Comparação de Técnicas de aprendizado supervisionado
##------------------------------------------------------------------------

# Bibliotecas padrão
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Carregando os dados
dataset = pd.read_csv('Case_cobranca.csv') 

#------------------------------------------------------------------------------------------
# Pré-processamento das variáveis
#------------------------------------------------------------------------------------------
## Tratamento de nulos no alvo --- Tempo de Atraso - transformação para alvo binário (>90 dias) 
dataset['ALVO']   = [0 if np.isnan(x) or x > 90 else 1 for x in dataset['TEMP_RECUPERACAO']]
## Tratamento de nulos e normalização --- Variáveis de entrada numéricas
dataset['PRE_IDADE']        = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # Trata mínimo
dataset['PRE_IDADE']        = [1. if x > 76 else (x-18)/(76-18) for x in dataset['PRE_IDADE']] # Trata máximo por percentil 99 e coloca na fórmula
dataset['PRE_QTDE_DIVIDAS'] = [0.  if np.isnan(x) else x/16. for x in dataset['QTD_DIVIDAS']] # retirada de outlier com percentil 99 e normalização     
##--- Dummies - transformação de atributos categóricos em numéricos e tratamanto de nulos ---------------
dataset['PRE_NOVO']         = [1 if x=='NOVO'                      else 0 for x in dataset['TIPO_CLIENTE']]    
dataset['PRE_TOMADOR_VAZIO']= [1 if x=='TOMADOR' or str(x)=='nan'  else 0 for x in dataset['TIPO_CLIENTE']]                        
dataset['PRE_CDC']          = [1 if x=='CDC'                       else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_PESSOAL']      = [1 if x=='PESSOAL'                   else 0 for x in dataset['TIPO_EMPRESTIMO']]
dataset['PRE_SEXO_M']       = [1 if x=='M'                         else 0 for x in dataset['CD_SEXO']]


##------------------------------------------------------------
## Separando em dados de treinamento e teste
##------------------------------------------------------------
y = dataset['ALVO']              # Carrega alvo ou dataset.iloc[:,7].values
X = dataset.iloc[:, 8:15].values # Carrega colunas 8, 9, 10, 11, 12, 13 e 14 (a 15 não existe até este momento)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 25)

#---------------------------------------------------------------------------
## Calculando a KNN - Aprendizado supervisionado  
#---------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

gx = []
gy = []

# Para k=1
Classifier_kNN = KNeighborsClassifier(n_neighbors=1, weights='uniform', algorithm='brute', p=2)
Classifier_kNN.fit(X_train, y_train)
y_pred_test_KNN    = Classifier_kNN.predict(X_test)
Erro_KNN_Classificacao = np.mean(np.absolute(y_pred_test_KNN - y_test))
print('---------------------------------------------------------------')
print('k', 'Erro de Classificação')
print('1',Erro_KNN_Classificacao)
gx.append(1)
gy.append(Erro_KNN_Classificacao)


## Loop para achar o melhor k
for k in range(5, 201, 5):
    Classifier_kNN = KNeighborsClassifier(n_neighbors=k, weights='uniform', algorithm='brute', p=2)
    Classifier_kNN.fit(X_train, y_train)
    y_pred_test_KNN    = Classifier_kNN.predict(X_test)

    ## Cálculo dos erros da classificação e Matriz de confusão da RNA
    Erro_KNN_Classificacao = np.mean(np.absolute(y_pred_test_KNN - y_test))
    print(k,Erro_KNN_Classificacao)
    
    gx.append(k)
    gy.append(Erro_KNN_Classificacao)
print('---------------------------------------------------------------')


print('------------------------- Gráfico -----------------------------')
print()
plt.plot(gx,gy)
plt.plot(gx,gy, 'bo') 
plt.title('Escolha do Melhor k')
plt.ylabel('Erro de Classificação')
plt.xlabel('Valor de k')
plt.show()





