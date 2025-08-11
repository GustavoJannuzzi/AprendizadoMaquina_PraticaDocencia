import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Função para gerar EDA (Exploratory Data Analysis)
def generate_eda(df):
    # Histograma
    st.subheader("Histograma das variáveis numéricas")
    df.hist(bins=30, figsize=(10, 8))
    st.pyplot()

    # Boxplot
    st.subheader("Boxplot das variáveis numéricas")
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df[col])
        st.pyplot()

    # Correlação
    st.subheader("Matriz de Correlação")
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.2f', linewidths=0.5)
    st.pyplot()

    # Pairplot
    st.subheader("Pairplot das variáveis numéricas")
    sns.pairplot(df.select_dtypes(include=['float64', 'int64']))
    st.pyplot()

# Função de pré-processamento
def process_data(df, features, target, test_size=0.2):
    X = df[features]
    y = df[target]
    return X, y
