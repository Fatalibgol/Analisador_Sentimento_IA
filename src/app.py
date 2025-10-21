# Arquivo: src/app.py

from flask import Flask, request, jsonify
import joblib
import re
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import nltk

# 1. Carregar o Modelo e o Vetorizador (o coração da nossa I.A.)
try:
    MODEL_PATH = 'models/modelo_sentimento.pkl'
    VECTORIZER_PATH = 'models/vetorizador.pkl'
    
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    stopwords_portugues = set(stopwords.words('portuguese'))
    tokenizer = RegexpTokenizer(r'\w+')
    print("Modelo e Vetorizador carregados com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar o modelo de I.A. Verifique se os arquivos .pkl estão em 'models/'. Erro: {e}")
    # Retornamos para que o app não inicie sem o modelo
    exit()

# 2. Inicializar o aplicativo Flask (o servidor web)
app = Flask(__name__)

# 3. Função de Pré-Processamento (igual ao data_processor, mas só com uma frase)
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = tokenizer.tokenize(text)
    tokens_filtrados = [palavra for palavra in tokens if palavra not in stopwords_portugues]
    return " ".join(tokens_filtrados)

# 4. Definição da Rota (Endpoint) da API
@app.route('/predict', methods=['POST'])
def predict():
    # A sintaxe request.json pega os dados enviados via POST (JSON)
    data = request.json
    
    # Validação básica
    if 'comentario' not in data:
        return jsonify({'error': 'Campo "comentario" ausente no JSON'}), 400

    comentario = data['comentario']
    
    # Processamento e Vetorização
    comentario_limpo = preprocess_text(comentario)
    
    # Vetoriza o comentário limpo
    comentario_vec = vectorizer.transform([comentario_limpo])
    
    # Previsão da I.A.
    prediction = model.predict(comentario_vec)[0]
    
    # Retorna o resultado em formato JSON
    return jsonify({
        'comentario_original': comentario,
        'sentimento_previsto': str(prediction)
    })

# 5. Inicialização do Servidor
if __name__ == '__main__':
    # Usamos o host '0.0.0.0' para que o Azure (ou o Docker) possa acessá-lo
    app.run(host='0.0.0.0', port=5000, debug=True)