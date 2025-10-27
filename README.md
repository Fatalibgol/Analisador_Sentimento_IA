🧠 Pipeline MLOps de Análise de Sentimento (Português)

🎯 Visão Geral do Projeto

Status: Completo e Pronto para Deploy em Produção (Azure App Service)

Este projeto demonstra a criação de um Pipeline MLOps (Machine Learning Operations) completo, transformando avaliações de texto não estruturado em dados classificáveis (Sentimento 1-5). O foco é provar a proficiência em levar soluções de Inteligência Artificial do notebook para a nuvem.

Desenvolvido por: [Seu Nome Completo]
Foco de Carreira: Engenharia de Software, Analista Júnior / Ciência de Dados

🛠️ Stack Tecnológico

Categoria

Tecnologia

Finalidade

Linguagem Principal

Python 3.10+

Lógica e Processamento de Dados.

I.A. / ML

Scikit-learn, joblib

Treinamento, Avaliação e Serialização do Modelo de Classificação.

NLP

NLTK (Tokenização, Stopwords), RegEx

Pré-processamento de texto em Português.

Backend / API

Flask, Gunicorn

Criação de um endpoint RESTful para consumo em tempo real.

Frontend / Demo

Streamlit

Interface interativa amigável para demonstração.

Infraestrutura / MLOps

Docker, Azure App Service, Azure CLI

Containerização e Deployment em Ambiente de Produção.

🚀 Estrutura e Etapas do Pipeline

O projeto segue a estrutura padrão de MLOps, separando as responsabilidades:

1. Engenharia de Dados (Análise)

Script: src/data_processor.py

Função: Lê o CSV (separador ;, codificação latin-1), trata valores nulos e realiza a limpeza de texto (minúsculas, remoção de pontuação e stopwords). O resultado é salvo em data/processed/dados_limpos.csv.

2. Machine Learning (Treinamento da I.A.)

Script: src/model_trainer.py

Função: Carrega os dados limpos, usa TF-IDF para vetorização e treina o modelo Logistic Regression. Os modelos finais (modelo_sentimento.pkl e vetorizador.pkl) são salvos na pasta models/.

Diagnóstico: A baixa acurácia inicial do modelo (bias) é intencional para fins de demonstração, provando que o Engenheiro de Software/Analista consegue diagnosticar e planejar o próximo passo: a Otimização com um dataset maior.

3. Serviço e Demonstração (Interfaces)

src/app.py: A API de Produção (Flask). É o endpoint (/predict) que o Azure App Service irá rodar via Gunicorn.

src/interface.py: A Interface Web (Streamlit) para análise interativa de texto único e classificação de lote (upload de CSV).

☁️ Instruções para Deploy no Azure

O projeto está configurado para o Azure App Service (Web App for Containers), demonstrando conhecimento em ambientes de produção.

Pré-requisitos

Conta Azure ativa.

Azure CLI instalado e logado (az login).

Um Personal Access Token (PAT) do GitHub para o deploy.

⚙️ Comandos de Execução Local

Ativar Ambiente: conda activate analise_ia

Preparar Dados: python src/data_processor.py

Treinar I.A.: python src/model_trainer.py

Rodar a Interface Web (Demo): streamlit run src/interface.py (Abre http://localhost:8501)

🐳 Arquivos de Infraestrutura para o Deploy

Dockerfile: Contém as instruções para construir a imagem Docker do projeto.

requirements.txt: Lista todas as dependências (flask, gunicorn, scikit-learn).

gunicorn_conf.py: Configuração do servidor de produção (gunicorn) na porta 8000.
