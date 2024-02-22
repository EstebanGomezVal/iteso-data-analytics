# Instalar librerias 

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.title("Visualización del Titanic")
st.header("Data set")
st.markdown("*Este es una base de datos acerca del Titanic*")

# Importar csv
data = ("Titanic-Dataset.csv")
df = pd.read_csv(data)
df

st.markdown("**A partir de esta base de datos, se concluyeron las gráficas a continuación**")
st.markdown("Al Titanic abordaron 891 pasajeros de los cuales  342 sobrevivieron y 549 fallecieron")

fig1, ax = plt.subplots()
df["Survived"].plot(kind = "hist", bins = 2, rwidth = 0.7, color = (.13, .02, .24) )
plt.xticks([.25, 0.75], ["Muertos", "Vivos"])
plt.close(fig1)

st.pyplot(fig1)

st.header("Clases")
st.markdown("491 personas estuvieron en la clase 3")
st.markdown("216 abordaron a la calse 1")
st.markdown("184 en la clase 2")

fig2, ax = plt.subplots()
plt.ylabel("Personas")
df["Pclass"].plot(kind = "hist", bins = 3, rwidth = 0.7, color = (.45, .84, .12) )
plt.xticks([1.33, 2, 2.65], ["Clase 1", "Clase 2", "Clase 3"])
plt.ylabel("Personas")
plt.close(fig2)

st.pyplot(fig2)

hombre = 0
mujer = 0

for i in df["Sex"]:
    if i == "female":
        mujer += 1
    else:
        hombre += 1

st.header("Género")

st.markdown("De los 891 pasajeros del Titanic abordaron 314 mujeres y 577 hombres")

fig3, ax = plt.subplots()
plt.bar(["mujer", "hombre"], [mujer, hombre], color = ['pink', 'blue'])
plt.close(fig3)

st.pyplot(fig3)

st.header("Número de hermanos de cada pasajero")
st.markdown("En el Titanic 608 pasajeros no tenian hermanos, mientras que 283 contaban con uno o más")

fig4, ax = plt.subplots()
df["SibSp"].plot(kind = "hist", rwidth = 0.9, color = [0.4, 0.34, 0.7])
plt.xlabel("Número de hermanos")
plt.ylabel("Personas")
plt.close(fig4)

st.pyplot(fig4)

st.header("Cantidad de familiares que acompañaban a cada pasajero")
st.markdown("En esta gráfica se cuentan los familiares que cada pasajero tenia estando en el Titanic")

fig5, ax = plt.subplots()
df["Parch"].plot(kind = "hist", rwidth = 0.9, color = "y")
plt.xlabel("Familiares que los acompañaban")
plt.ylabel("Personas")
plt.close(fig5)

st.pyplot(fig5)

st.header("Embarcación")
st.markdown("Del Titanic 644 pasajeros venian de Southampton, 77 pasajeros venian de Queenstown y 168 pasajeros de Cherbourg")

fig6, ax = plt.subplots()
df["Embarked"] = df["Embarked"].map({'S': 0, 'Q': 1, 'C':2})
df['Embarked'].plot(kind = 'hist', bins = 3 ,rwidth = 0.9, color = [0.62, 0.12, 0.73])
plt.xticks([0.35, 1, 1.7], ["Southampton", 'Queenstown', 'Cherbourg'])
plt.ylabel("Personas")

plt.close(fig6)

st.pyplot(fig6)

st.header("Edad")
st.markdown("La edad del pasajero más grande fue de 80 años mientras que el más pequeño fue de 0.42 años")