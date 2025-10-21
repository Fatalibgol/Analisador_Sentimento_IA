import streamlit as st
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk
import io 

# --- CONFIGURAÇÃO DA I.A. (Carrega o modelo) ---
MODEL_PATH = 'models/modelo_sentimento.pkl'
VECTORIZER_PATH = 'models/vetorizador.pkl'
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    stopwords_portugues = set(stopwords.words('portuguese'))
    tokenizer = RegexpTokenizer(r'\w+')
except Exception:
    st.error("ERRO: Modelos de I.A. (.pkl) não encontrados. Execute 'python src/model_trainer.py' primeiro.")
    st.stop()


# Função de Pré-Processamento (Comum às duas seções)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords_portugues]
    return " ".join(tokens_filtrados)

# Função para Mapear o Score (para visualização)
def map_score_to_emoji(score_str):
    try:
        score = int(score_str)
        # LÓGICA ATUALIZADA: 5 é Positivo, 3 e 4 são Neutros, <= 2 é Negativo
        if score == 5:
            return f"⭐{score} (Positivo)", "success"
        elif score == 3 or score == 4: # Agora 3 E 4 são considerados Neutros
            return f"🟡{score} (Neutro)", "warning"
        else: # 1 e 2 (ou menos)
            return f"🔴{score} (Negativo/Crítico)", "error"
    except ValueError:
        return f"{score_str} (Erro)", "error"


# --- CONFIGURAÇÃO DA INTERFACE STREAMLIT ---
st.set_page_config(page_title="Analisador de Sentimento (I.A. em Português)", layout="wide")

st.title("Sistema de Classificação de Sentimento")
st.markdown("Use sua I.A. para classificar comentários de 1 (Negativo) a 5 (Positivo).")

# --- SEÇÃO 1: Análise de Texto Único ---
st.header("1. Análise de Texto Único")
comentario_usuario = st.text_area("Insira o Comentário de Cliente Aqui:", 
                                  "O produto chegou no prazo e a qualidade é muito boa. Ficarei de olho em mais ofertas!")

if st.button("Analisar Sentimento (Rodar I.A.)"):
    if comentario_usuario:
        with st.spinner('Analisando o comentário...'):
            sentimento_previsto = str(model.predict(vectorizer.transform([preprocess_text(comentario_usuario)]))[0])
        
        resultado_texto, cor = map_score_to_emoji(sentimento_previsto)

        st.subheader("Resultado da Classificação")
        st.info(f"**Sentimento Previsto:** {resultado_texto}")
        
        st.caption("Detalhes da I.A.")
        st.code(f"Texto Limpo (Vetorizado): {preprocess_text(comentario_usuario)}")
    else:
        st.warning("Por favor, insira um comentário para análise.")


# --- SEÇÃO 2: Upload e Classificação de Arquivo CSV (Com Seletor de Coluna) ---
st.header("2. Classificação de Lote (Upload CSV)")
uploaded_file = st.file_uploader("Faça upload de um arquivo CSV (Sep. ';') para Classificar", type="csv")

if uploaded_file is not None:
    try:
        # Lendo o arquivo CSV com a correção de codificação
        df_lote = pd.read_csv(uploaded_file, sep=';', encoding='latin-1') 
        
        # Sugestão de coluna (a que contiver mais texto livre)
        text_cols = [col for col in df_lote.columns if df_lote[col].dtype == 'object' and df_lote[col].str.len().mean() > 5]
        
        # Seletor de Coluna (O usuário escolhe)
        COLUNA_TEXTO_LOTE = st.selectbox(
            "Selecione a coluna que contém o COMENTÁRIO:", 
            options=df_lote.columns.tolist(),
            index=df_lote.columns.get_loc(text_cols[0]) if text_cols else 0
        )

        st.success(f"Arquivo carregado com sucesso. Total de {len(df_lote)} linhas.")
        st.caption(f"A I.A. irá classificar a coluna: **{COLUNA_TEXTO_LOTE}**")
        
        # Botão de Classificação de Lote
        if st.button(f"Classificar {len(df_lote)} Comentários (Rodar I.A. em Lote)"):
            if COLUNA_TEXTO_LOTE not in df_lote.columns:
                st.error("Coluna selecionada não existe no arquivo.")
            else:
                with st.spinner('Classificando todas as linhas. Isso pode levar um momento...'):
                    
                    # 1. Pré-processa a coluna de texto inteira
                    df_lote['Texto_Processado'] = df_lote[COLUNA_TEXTO_LOTE].apply(preprocess_text)
                    
                    # 2. Vetoriza e Previsão
                    X_lote_vec = vectorizer.transform(df_lote['Texto_Processado'])
                    df_lote['Previsao_Sentimento'] = model.predict(X_lote_vec)
                    
                    # 3. Mapeia para texto/emoji
                    df_lote['Resultado'] = df_lote['Previsao_Sentimento'].astype(str).apply(lambda x: map_score_to_emoji(x)[0])
                    
                    st.subheader("Resultados da Classificação por Lote")
                    st.dataframe(df_lote[[COLUNA_TEXTO_LOTE, 'Resultado', 'Previsao_Sentimento']])
                    
                    # Permite o download do arquivo classificado
                    csv_saida = df_lote.to_csv(index=False).encode('utf-8') 
                    st.download_button(
                        label="Baixar CSV Classificado",
                        data=csv_saida,
                        file_name='comentarios_classificados.csv',
                        mime='text/csv',
                    )
    
    except Exception as e:
        st.error(f"Erro ao ler o arquivo. Verifique se o formato está correto (sep=';'). Erro: {e}")
