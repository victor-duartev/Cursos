#Importação de pandas
import pandas as pd

#Leitura dos data bases
base = pd.read_csv('census.csv')
#Divisão em data frames previsores e classe 
previsores = base.iloc[:,0:14].values
classe = base.iloc[:,14].values

#Importação de encoders
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Importação de transformador de colunas
from sklearn.compose import ColumnTransformer

#Criação de variável para ColumnTransformer com parâmetros já setados
column_tranformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), 
                                        [1, 3, 5, 6, 7, 8, 9, 13])],remainder='passthrough')
#Atualização de df previsores com ColumnTransformer aplicado
previsores = column_tranformer.fit_transform(previsores).toarray()

#Criação de variável de LabelEncoder para df classe
labelencoder_classe = LabelEncoder()
#Atualização de df classe com LabelEncoder aplicado
classe = labelencoder_classe.fit_transform(classe)

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

#Importação do KNN
from sklearn.neighbors import KNeighborsClassifier
#Definição de variável para KNN
classificador = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
#Aplicação do NB nos df's previsores e classe
classificador.fit(previsores_treinamento, classe_treinamento)
#Criação de variável com resultados do KNN
previsoes = classificador.predict(previsores_teste)

#Importação de métricas
from sklearn.metrics import confusion_matrix, accuracy_score
#Criação variável para precisão do algoritimo
precisao = accuracy_score(classe_teste, previsoes) 
#Criação variável para matriz de confusão
matriz = confusion_matrix(classe_teste, previsoes)
