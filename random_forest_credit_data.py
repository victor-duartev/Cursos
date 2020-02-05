#Importação de pandas
import pandas as pd
#Importação de numpy
import numpy as np

#Leitura dos data bases
base = pd.read_csv('credit-data.csv')
#Substituição de valores negativos para média das idades
base.loc[base.age < 0, 'age'] = 40.92
              
#Divisão em dataframes previsores e classe 
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#Importação do SimpleImputer
from sklearn.impute import SimpleImputer
#Definição de variável para imputer com parâmetros setados
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
#Aplicação do imputer nas respectivas colunas
imputer = imputer.fit(previsores[:, 1:4])
#Atualização do df previsores com imputer aplicado
previsores[:, 1:4] = imputer.transform(previsores[:, 1:4])

#Importação de padronizador
from sklearn.preprocessing import StandardScaler
#Atribuição de padronizador
scaler = StandardScaler()
#Atualização do df previsores com padronizador aplicado
previsores = scaler.fit_transform(previsores)

#Importação de divisor train-test
from sklearn.model_selection import train_test_split
#Divisão do df em treinamento e teste com parâmetros setados
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.15, random_state=0)

#Importação do random forest
from sklearn.ensemble import RandomForestClassifier
#Definição de variável para random forest
classificador = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
#Aplicação nos df's previsores e classe
classificador.fit(previsores_treinamento, classe_treinamento)
#Criação de variável com resultados
previsoes = classificador.predict(previsores_teste)

#Importação de métricas
from sklearn.metrics import confusion_matrix, accuracy_score
#Criação variável para precisão do algoritimo
precisao = accuracy_score(classe_teste, previsoes) 
#Criação variável para matriz de confusão
matriz = confusion_matrix(classe_teste, previsoes)