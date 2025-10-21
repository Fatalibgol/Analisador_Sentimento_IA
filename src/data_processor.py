# Arquivo: src/data_processor.py

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
# Importamos o tokenizer mais seguro que não depende de download extra do punkt
from nltk.tokenize import RegexpTokenizer 
import os # ESSENCIAL: Para criar a pasta de saída!
import sys 


# --- CAMINHOS ---
INPUT_PATH = 'data/raw/dataset_avaliacoes1.csv' 
OUTPUT_PATH = 'data/processed/dados_limpos.csv'

# --- NOMES DE COLUNAS ---
COLUNA_TEXTO = 'comentario'       
COLUNA_SENTIMENTO = 'classificacao' 


def process_data():
    print("Iniciando o carregamento dos dados...")
    
    # 1. TENTATIVA DE CARREGAMENTO (USANDO sep=';')
    try:
        df = pd.read_csv(INPUT_PATH, sep=';') 
        print(f"Dados carregados com sucesso. Total de {len(df)} linhas.")
    except Exception as e:
        print(f"ERRO DE CARREGAMENTO: {e}. Verifique se o separador ('sep=;') está correto e se o arquivo existe.")
        sys.exit(1) 
    
    # === GARANTIR O DIRETÓRIO DE SAÍDA ===
    # Esta é a lógica que resolve o último OSError
    parent_dir = os.path.dirname(OUTPUT_PATH)
    if parent_dir and not os.path.exists(parent_dir):
        # Cria o diretório (e os diretórios pais, se necessário)
        os.makedirs(parent_dir, exist_ok=True)
        print(f"Diretório criado: {parent_dir}")
    # ====================================

    # --- Etapa 1: Tratamento de Nulos ---
    print("\nVerificando e tratando valores nulos...")
    
    # 1.1 Remover linhas onde o texto ou sentimento está vazio/nulo
    df.dropna(subset=[COLUNA_TEXTO, COLUNA_SENTIMENTO], inplace=True)
    
    # 1.2 Remover linhas onde o texto é apenas espaços vazios (garante que não há problemas de tipo)
    df = df[(df[COLUNA_TEXTO].astype(str) != '') & (df[COLUNA_TEXTO].astype(str).str.strip().astype(bool))]
    print(f"Linhas restantes após remoção de nulos/vazios: {len(df)}")


    # --- Etapa 2: Limpeza e Padronização de Texto ---
    print("\nIniciando a limpeza e padronização do texto...")
    
    # 2.1 Converter para minúsculas
    df[COLUNA_TEXTO] = df[COLUNA_TEXTO].astype(str).str.lower()
    
    # 2.2 Remoção de pontuação e caracteres especiais 
    df[COLUNA_TEXTO] = df[COLUNA_TEXTO].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
    
    # 2.3 Remover espaços extras (múltiplos espaços por um único espaço)
    df[COLUNA_TEXTO] = df[COLUNA_TEXTO].str.replace(r'\s+', ' ', regex=True)
    
    
    # --- Etapa 3: Tokenização e Remoção de Stopwords (Solução Robusta) ---
    print("Iniciando Tokenização e remoção de Stopwords...")
    stopwords_portugues = set(stopwords.words('portuguese'))
    
    tokenizer = RegexpTokenizer(r'\w+') 

    def remover_stopwords(texto):
        # 1. Tokenização segura (apenas palavras)
        tokens = tokenizer.tokenize(texto)
        
        # 2. Filtragem de Stopwords
        tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords_portugues]
        
        # 3. Junta as palavras de volta
        return " ".join(tokens_filtrados)

    # 3.1 Aplicar a função à coluna e criar a nova coluna processada
    df['Texto_Processado'] = df[COLUNA_TEXTO].apply(remover_stopwords)
    
    print("Limpeza e Tokenização concluídas.")
    print("\nAmostra da transformação (original vs. processado):")
    # Limita a exibição para as 5 primeiras linhas
    print(df[[COLUNA_TEXTO, 'Texto_Processado']].head())

    # 4. Salvar os dados processados na pasta 'data/processed'
    df[[COLUNA_SENTIMENTO, 'Texto_Processado']].to_csv(OUTPUT_PATH, index=False)
    print(f"\nDados processados e salvos em {OUTPUT_PATH}")


# Ponto de entrada do script
if __name__ == '__main__':
    process_data()