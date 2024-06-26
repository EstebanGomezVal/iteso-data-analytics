import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("Modelo predictivo de problemas cardiacos")
st.header("Definición del Problema")
st.markdown("*El propósito de este proyecto es desarrollar un programa para la creación de un modelo predictivo acerca de personas que poseen enfermedad de corazón. En base a ciertas características, el modelo entrenado será capaz de predecir qué personas son probables de adquirir una enfermedad cardíaca, basandose en 13 distintos “features” tales como azúcar en la sangre, anginas inducidas, presión arterial del paciente, entre otras buscando relación entre estas y la posesión de una enfermedad cardiáca.*")
st.markdown("*Este proyecto se centra en el análisis de datos de múltiples pacientes de diversas edades con el objetivo de reducir los problemas cardíacos en las personas. Se busca analizar varios features del paciente para predecir la probabilidad de que adquiera un problema cardíaco en el futuro, basándose en casos pasados de pacientes con características similares que han sido diagnosticados con enfermedades cardíacas. El propósito principal es concienciar a los pacientes a través de casos pasados para buscar que cuiden su salud y reducir la probabilidad de adquirir problemas cardíacos.*")

data_path = "src/Streamlit/heart_2.csv"
df = pd.read_csv(data_path)
st.write(df)

st.markdown("Este DataFrame cuenta con 1025 filas las cuales representan a distintos pacientes y 14 columnas las cuales representan 14 diferentes features de dichos pacientes, las cuales son:")

st.markdown('- Edad')

st.markdown('- Sexo')

st.markdown('- Tipo de dolor en el pecho (4 valores):')
st.markdown('+++++ Valor 0: Angina típica')
st.markdown('+++++ Valor 1: Angina atípica')
st.markdown('+++++ Valor 2: Sin dolor de angina')
st.markdown('+++++ Valor 3: Asintomático')

st.markdown('- Presión arterial en reposo medido en milímetros de mercurio al ingresar al hospital')

st.markdown('- Colesterol sérico en mg/dl')

st.markdown('- Azúcar en sangre en ayunas > 120 mg/dl:')
st.markdown('+++++ Valor 0: Falso')
st.markdown('+++++ Valor 1: Verdadero')

st.markdown('- Resultados electrocardiográficos en reposo:')
st.markdown('+++++ Valor 0: Normal')
st.markdown('+++++ Valor 1: Poseer anomalía de la onda ST-T')
st.markdown('+++++ Valor 2: Poseer hipertrofia ventricular izquierda probable o definitiva')

st.markdown('- Frecuencia cardíaca máxima alcanzada')

st.markdown('- Angina inducida por ejercicio:')
st.markdown('+++++ Valor 0: Falso')
st.markdown('+++++ Valor 1: Verdadero')

st.markdown('- Depresión del segmento ST inducida por ejercicio en relación al reposo')

st.markdown('- La pendiente del segmento ST del ejercicio máximo:')
st.markdown('+++++ Valor 0: Pendiente ascendente')
st.markdown('+++++ Valor 1: Pendiente plana')
st.markdown('+++++ Valor 2: Pendiente descendente')

st.markdown('- Número de vasos principales (0-3) coloreados por fluoroscopia')

st.markdown('- Thal: ')
st.markdown('+++++ Valor 0: Normal')
st.markdown('+++++ Valor 1: Defecto fijo')
st.markdown('+++++ Valor 2: Defecto reversible')

st.markdown('- Target:')
st.markdown('+++++ Valor 0: Menor riesgo de ataque al corazón')
st.markdown('+++++ Valor 1: Mayor riesgo de ataque al corazón')

st.markdown('Utilicé el modelo de clasificación ya que en mi “target” existen dos valores (binario), los que poseen y los que no poseen problemas de corazón, por esta razón utilicé este modelo, para mediante características conocer a que clase podría pertenecer un nuevo paciente.')

st.header("Descripción del DataFrame")

df_summary = df.describe()
st.write(df_summary)

st.header("Visualización de la distribución de datos")
cols = st.selectbox('Elige la columna', df.columns)
fig, ax = plt.subplots()
ax.hist(df[cols], bins = 20, alpha = 0.7, color='blue')
ax.set_title(f'{cols} distribution')
ax.set_xlabel('Valores')
ax.set_ylabel('Frecuencia')
st.pyplot(fig)

st.subheader("**Target**")
st.markdown('En este DataFrame, 526 personas poseen riesgo de un problema cardíaco, mientras que las otras 499 personas no poseen riesgo de un problema cardíaco.')
st.write(df['target'].value_counts())
fig, ax = plt.subplots()
ax.pie(df['target'].value_counts(), labels=["Problema cardiaco", "Sin problema cardiaco"], colors=['blue', (0.1, 0.6, 0.8)])
ax.set_title("Target")
st.pyplot(fig)

st.subheader("**Sexo**")
st.markdown('En este DataFrame, entre los encuestados se encuentran 713 hombres y 312 mujeres .')
st.write(df['sex'].value_counts())
fig, ax = plt.subplots()
ax.pie(df['sex'].value_counts(), labels=["Hombre", "Mujer"], colors=['blue', (0.1, 0.6, 0.8)])
ax.set_title("Sexo")
st.pyplot(fig)

st.subheader('**Dolor de pecho**')
st.markdown('Aquí podemos observar los tipos de dolor de angina que poseen los entrevistados.')
st.write(df['cp'].value_counts())
fig, ax = plt.subplots()
sns.countplot(x=df['cp'], color=(0.4, 0.6, 0.8))
plt.title("Cp")
plt.xticks([0, 1, 2, 3], ['Angina típica', 'Angina atípica', 'Sin dolor de angina', 'Asintomático'])
st.pyplot()

st.subheader('**Dolor de pecho contra target**')
st.markdown('Se puede observar como las personas con angina típica son la mayoria pacientes sin riesgo, mientras que en personas sin dolor de angina, angina atípica y asintomáticos, se puede observar como los pacientes con riesgo son mayoría.')
sns.countplot(x = df['cp'], hue=df['target'].astype(str))
plt.legend(labels=['Sin riesgo', 'Con riesgo'])
plt.xticks([0, 1, 2, 3], ['Angina típica', 'Angina atípica', 'Sin dolor de angina', 'Asintomático'])
st.pyplot()

st.subheader('**Azúcar en sangre en ayunas > 120 mg/dl contra target**')
st.markdown('En esta gráfica se puede observar que la mayoría de pacientes no sobrepasan los 120 mg/dl de azúcar en la sangre, sin embargo se puede observar como de los pacientes que tienen menos de 120 mg/dl azúcar en la sangre, existen más pacientes propensos a riesgo de un problema de corazón.')
sns.countplot(x = df['fbs'], hue = df['target'].astype(str))
plt.legend(labels=['Sin riesgo', 'Con riesgo'])
plt.xticks([0, 1], ['Menor', 'Mayor'])
st.pyplot()

st.header('Revisión de datos atípicos')
st.markdown('Una breve visualización de las variables numéricas con el propósito de observar valores atípicos en dichas variables usando el método de caja de bigotes')

numericas = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']
df_numerico = df.loc[:, numericas].copy()

fig = plt.figure()
for i in range(len(df_numerico.columns)):
    plt.subplot(2, 4, i + 1)
    sns.boxplot(df_numerico[df_numerico.columns[i]], color='blue')
    plt.title(df_numerico.columns[i])
st.pyplot(fig)

st.header('Correlaciones')
st.markdown('Una breve visualización de un mapa de calor para observar variables que tengan relaciones entre sí')

fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df.corr(), linewidth=.01, annot=True, cmap="winter", xticklabels=True, yticklabels=True, ax=ax)
st.pyplot(fig)

st.markdown('Se puede observar que existe una correlación entre thalach y slope, con esta observación podemos graficar con el fin de obtener conclusiones')
X1 = df['thalach'].to_numpy()
X2 = df['slope'].to_numpy()
y1 = df['target'].to_numpy()
fig, ax = plt.subplots()
scatter1 = ax.scatter(X2[y1 == 0], X1[y1 == 0], label='Sin problema cardiaco', color='blue')
scatter2 = ax.scatter(X2[y1 == 1], X1[y1 == 1], label='Con problema cardiaco', color=(0.1, 0.6, 0.8))
ax.legend()
st.pyplot(fig)

st.markdown('Somos capaces de observar con un mayor thalach, mayor probabilidad existe de que la persona adquiera un problema cardíaco, se puede observar también que con pendiente normal exsiten más personas sin problemas cardíacos, mientras que con una pendiente descendiente existen más personas con probemas cardíacos.')

st.header('Normalización de datos para una predicción')
st.markdown('Se utilizaron varios métodos como get dummies y standard scaler con el fin de obtener una mejor predicción acerca de que tan probable es que una perosna adquiera o no un problema cardíaco')

col_categ = ['cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
col_num = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

df = pd.get_dummies(df, columns = col_categ, drop_first = True).astype(int)

SS = StandardScaler()
df[col_num] = SS.fit_transform(df[col_num])

st.write(df)

X = df.drop('target', axis = 1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2, random_state = 32)

st.markdown('A partir de ello se utilizaron métodos con el fin de verificar que tan preciso es el modelo predictivo que se creó:')

st.markdown('Utilizando Linear Regression se obtuvo una calificación de 0.6139830639438726')
st.markdown('Utilizando KNeighbors Classifier se obtuvo una calificación de 0.8682926829268293')
st.markdown('Utilizando Logistic Regression se obtuvo una calificación de 0.8829268292682927')

st.subheader('KMeans')
st.markdown('Mediante la gráfica de codo, buscamos la mejor cantidad de clusters para el DataFrame, en este caso utilizaremos 8')

lista = []

for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, random_state=32)
    kmeans.fit(X)
    lista.append(kmeans.inertia_)
plt.figure()
plt.plot(range(1,20), lista, marker="x")
plt.grid()
st.pyplot()

kmeans = KMeans(n_clusters = 8, random_state = 32)
clusters = kmeans.fit_predict(df)

df1 = df.copy()
df1["cluster"] = clusters

centroides = kmeans.cluster_centers_

df2 = TSNE(n_components = 2, random_state = 35, perplexity = 8)
tsne_df = df2.fit_transform(df1)

df3 = TSNE(n_components = 2, random_state = 42, perplexity = 6)
df_centroides = df3.fit_transform(centroides)

st.markdown('Utilizando TSNE somos capaces de visualizar los clusters donde se distribuyen distintos tipos de pacientes')

scatter = plt.scatter(tsne_df[:, 0], tsne_df[:, 1], c = df1["cluster"], cmap="rainbow", label="Clusters")
plt.scatter(df_centroides[:,0], df_centroides[:,1])
handles, labels = scatter.legend_elements()
plt.legend(handles, labels, title = "Clusters")
st.pyplot()

st.header('Usos de librerias')
st.markdown('*⁃ La librería de pandas la cual fue utilizada para importar el archivo csv al jupyter notebook.*')
st.markdown('*⁃	Se usó numpy para crear arreglos que se pudieran comparar entre sí.*')
st.markdown('*⁃	Se interpretaron los datos mediante gráficas de barras, de pastel, mapas de calor y caja de bigotes con las librerías de Seaborn y Matplotlib.*')
st.markdown('*⁃	Se usó el método de “get dummies” para separar las variables categóricas.*')
st.markdown('*⁃	Se hizo uso del método de Standard Scaler para expresar las variables numéricas con z-score. *')

st.header('Conclusiones')
st.markdown('En conclusión, la creación de este proyecto ha demostrado la utilidad de técnicas de análisis de datos y modelado predictivo para abordar problemas de salud, en particular, la prevención de enfermedades cardíacas. Se ha logrado desarrollar un modelo predictivo capaz de identificar a las personas con mayor riesgo de padecer problemas cardíacos, lo que puede ser una herramienta valiosa para los profesionales de la salud y las personas preocupadas por su bienestar cardiovascular.')
st.markdown('El análisis exploratorio de reveló interesantes relaciones entre las variables y la presencia de enfermedades cardíacas, destacando la importancia de factores como la presión arterial, los niveles de colesterol y la edad en la predicción del riesgo cardiovascular.')
st.markdown('En última instancia, este proyecto es una pequeña probada de la capacidad de la ciencia de datos para generar conocimientos significativos y aplicables en el campo de la salud y de cualquier otro, con el potencial de mejorar la calidad de vida y reducir la carga de enfermedades cardiovasculares en la sociedad.')