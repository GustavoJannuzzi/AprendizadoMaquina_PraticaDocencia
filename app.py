import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from utils import generate_eda, process_data

# Função principal
def main():
    # Sidebar para navegação
    st.sidebar.title("Machine Learning Modeling Interface")
    page = st.sidebar.radio("Escolha a página", ["Modelos de Regressão", "Modelos de Classificação"])

    # Carregar dados
    file = st.sidebar.file_uploader("Faça o upload de um arquivo CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.write("Preview dos dados:", df.head())

        # Seleção de colunas preditoras e alvo
        features = st.multiselect("Escolha as colunas preditoras", df.columns)
        target = st.selectbox("Escolha a coluna alvo", df.columns)

        # Divisão treino e teste
        test_size = st.slider("Escolha a porcentagem de dados para teste", 0.1, 0.9, 0.2)

        # Preprocessamento de dados
        if st.button("Gerar EDA"):
            generate_eda(df)

        # Carregar e treinar modelos com base na página selecionada
        if page == "Modelos de Regressão":
            model_options = ["Árvore de Decisão", "Random Forest", "Regressão Linear"]
        else:
            model_options = ["Árvore de Decisão", "Naive Bayes", "Random Forest", "Regressão Logística"]

        models_to_train = st.multiselect("Escolha os modelos para treinar", model_options)

        # Pré-processamento
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Preenchimento de valores ausentes
        imputer = SimpleImputer(strategy="mean")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        # Padronização/normalização
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Treinamento dos modelos selecionados
        for model_name in models_to_train:
            if model_name == "Árvore de Decisão":
                model = DecisionTreeClassifier() if page == "Modelos de Classificação" else DecisionTreeRegressor()
            elif model_name == "Naive Bayes":
                model = GaussianNB()
            elif model_name == "Random Forest":
                model = RandomForestClassifier() if page == "Modelos de Classificação" else RandomForestRegressor()
            elif model_name == "Regressão Logística":
                model = LogisticRegression()
            elif model_name == "Regressão Linear":
                model = LinearRegression()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Exibir métricas de avaliação
            st.subheader(f"Métricas para {model_name}")
            if page == "Modelos de Classificação":
                st.write(f"Acurácia: {accuracy_score(y_test, y_pred)}")
                st.write(f"Precisão: {precision_score(y_test, y_pred, average='macro')}")
                st.write(f"Recall: {recall_score(y_test, y_pred, average='macro')}")
                st.write(f"F1-score: {f1_score(y_test, y_pred, average='macro')}")
                st.write("Matriz de Confusão:")
                st.write(confusion_matrix(y_test, y_pred))
            else:
                st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")
                st.write(f"MAE: {mean_absolute_error(y_test, y_pred)}")
                st.write(f"R²: {r2_score(y_test, y_pred)}")

            # Teste unitário do modelo treinado
            st.subheader(f"Teste unitário para {model_name}")
            test_data = {}
            for feature in features:
                test_data[feature] = st.number_input(f"Digite o valor de {feature}")
            test_data = np.array(list(test_data.values())).reshape(1, -1)
            test_data = scaler.transform(test_data)
            st.write(f"Predição: {model.predict(test_data)}")

if __name__ == "__main__":
    main()
