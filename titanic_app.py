import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Título
st.title("🛳️ Previsão de Sobrevivência no Titanic")

# Entradas do usuário/seleção dos dados
st.header("🔍 Informe os dados do passageiro:")

pclass = st.selectbox("Classe do bilhete (1 = 1ª classe, 3 = 3ª classe)", [1, 2, 3])
sex = st.selectbox("Sexo", ['male', 'female'])
age = st.slider("Idade", 0.0, 100.0, 30.0)
sibsp = st.number_input("Irmãos/cônjuges a bordo", min_value=0, max_value=10, value=0)
parch = st.number_input("Pais/filhos a bordo", min_value=0, max_value=10, value=0)
fare = st.number_input("Valor da passagem", min_value=0.0, value=30.0)
embarked = st.selectbox("Porto de embarque", ['S', 'C', 'Q'])

# Converter para DataFrame
input_dict = {
    'Pclass': [pclass],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Sex_male': [1 if sex == 'male' else 0],
    'Embarked_Q': [1 if embarked == 'Q' else 0],
    'Embarked_S': [1 if embarked == 'S' else 0],
}
entrada = pd.DataFrame(input_dict)

# dataset 
df = pd.read_csv('titanic_limpo.csv')
dados = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked', 'Survived']]
dados = pd.get_dummies(dados, columns=['Sex', 'Embarked'], drop_first=True)
X = dados.drop('Survived', axis=1)
y = dados['Survived']
modelo = DecisionTreeClassifier(max_depth=4, random_state=42)
modelo.fit(X, y)

# Fazer previsão
if st.button("🔮 Prever"):
    predicao = modelo.predict(entrada)[0]
    resultado = "✅ Sobreviveu!" if predicao == 1 else "❌ Não sobreviveu."
    st.subheader(f"Resultado: {resultado}")
