import requests
import json

# URL da sua API local
url = "http://127.0.0.1:5000/predict"

# Dados de teste (use um comentário do seu CSV, como o 0 ou 1)
payload = {
    "comentario": "O produto excelente, chegou antes do prazo e bem embalado."
}

# Enviando a requisição POST
response = requests.post(url, json=payload)

# Imprimindo o resultado
if response.status_code == 200:
    print("\n--- Resultado da I.A. ---")
    print(json.dumps(response.json(), indent=4))
else:
    print(f"Erro na requisição: {response.status_code}")