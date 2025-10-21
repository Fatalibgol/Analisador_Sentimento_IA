# Arquivo: src/model_trainer.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib 
import sys
import os # Para garantir que a pasta de modelos existe

# --- CAMINHOS ---
INPUT_PATH = 'data/processed/dados_limpos.csv' 
MODEL_PATH = 'models/modelo_sentimento.pkl'
VECTORIZER_PATH = 'models/vetorizador.pkl' 


def train_model():
    print("Iniciando carregamento dos dados limpos...")
    try:
        # Carrega o arquivo que o data_processor acabou de criar
        df = pd.read_csv(INPUT_PATH)
        # Assumimos que o CSV limpo tem as colunas 'classificacao' e 'Texto_Processado'
        X = df['Texto_Processado'].astype(str) # Recurso (Texto limpo)
        y = df['classificacao']             # Variável Target (rótulo)
        print(f"Dados limpos carregados. Total de {len(df)} amostras.")
    except Exception as e:
        print(f"ERRO: Não foi possível carregar o arquivo limpo. {e}")
        sys.exit(1)

    # 1. Divisão de Dados (Treino e Teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 2. Vetorização (Transformar Texto em Números)
    # TfidfVectorizer: Cria uma representação numérica das palavras
    vectorizer = TfidfVectorizer(max_features=5000)
    
    # Fit e Transform nos dados de TREINO. Apenas Transform nos dados de TESTE.
    X_train_vec = vectorizer.fit_transform(X_train) 
    X_test_vec = vectorizer.transform(X_test)
    
    print("Dados vetorizados. Iniciando treinamento...")

    # 3. Treinamento do Modelo (I.A. - Regressão Logística)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train) 
    
    # 4. Avaliação do Modelo
    y_pred = model.predict(X_test_vec)
    acuracia = accuracy_score(y_test, y_pred)
    
    print("\n=============================================")
    print("TREINAMENTO CONCLUÍDO")
    print(f"Acurácia do Modelo (Performance da I.A.): {acuracia:.4f}")
    # O Classification Report fornece métricas mais detalhadas
    # print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
    print("=============================================")


    # 5. Salvar Modelo e Vetorizador
    
    # Criar pasta 'models' se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Diretório 'models/' criado.")
        
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Modelo (I.A.) e Vetorizador salvos em: {MODEL_PATH}")


# Ponto de entrada do script
if __name__ == '__main__':
    train_model()