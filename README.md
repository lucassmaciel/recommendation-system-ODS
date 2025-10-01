# Sistema de Recomendação de Livros

Projeto 1 — Oficina de Desenvolvimento de Sistemas I  
Professor: Fábio

## Equipe
- Caio Jorge da Cunha Queiroz — 2315310028
- Lucas Maciel Gomes — 2315310014
- Izabella de Lima Catrinck — 2315310033

## Visão Geral

Este projeto desenvolve um sistema de recomendação de livros utilizando filtragem colaborativa baseada em similaridade (kNN).  
O foco é entregar uma aplicação clara, organizada e avaliável, com separação entre backend (API) e frontend (interface).

### Principais Funcionalidades
- Registro/simulação de avaliações de usuários.
- Geração de recomendações personalizadas de livros.
- Visualização dos resultados em uma interface simples.
- Avaliação de desempenho via métricas de acurácia.

## Estrutura de Diretórios

A organização do projeto foi feita para facilitar a manutenção e atender aos critérios de avaliação:

backend/  
Contém a API responsável pelo processamento da recomendação.
É aqui que o algoritmo de filtragem colaborativa (kNN) será implementado.
O backend também expõe endpoints REST (ex.: /recomendar) que serão consumidos pelo frontend.

frontend/  
Contém a interface desenvolvida em Streamlit.
Permite a entrada de avaliações, execução do algoritmo e exibição das recomendações de livros ao usuário.

processed-data/  
Diretório destinado ao armazenamento dos datasets tratados.
Inclui:
- O catálogo de livros (ID, título, categoria).
- As avaliações dos usuários (usuário, livro, nota).

metrics/  
Pasta com scripts voltados ao cálculo da acurácia do sistema.
Aqui será feita a divisão dos dados em treino e teste, geração de recomendações e comparação com o “gabarito”.
Calcula-se a métrica de precisão por usuário, testando as abordagens: Coseno e Pearson.
Para rodar basta executar o comando: python -m metrics.test_user 

Para teste de Acurácia trazemos aqui 2 exemplos de usuários, com 20 recomendações:

Para o usuário 11676, temos:
Coseno
## Acertos: 14 de 20
## Acurácia: 70.00%

Pearson
## Acertos: 10 de 20
## Acurácia: 50.00%

Para o usuário 104636, temos:

Coseno
## Acertos: 7 de 20
## Acurácia: 35.00%

Pearson
## Acertos: 8 de 20
## Acurácia: 40.00%

README.md  
Documento de apresentação do projeto, contendo descrição, explicação da organização e instruções de execução.


## Instalação e Configuração

Este projeto utiliza o "uv" para gerenciamento de dependências. É necessário instalá-lo no seu computador:

### Instalação do UV

**Windows:**
```
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
``` 

**MacOS | Linux:**
```
curl -LsSf https://astral.sh/uv/install.sh | sh
``` 

### Configuração do Ambiente

Após instalar o UV, execute o comando abaixo na raiz do projeto:
```
uv sync -- frozen
``` 

Este comando irá:
- Criar o ambiente virtual (.venv)
- Instalar a versão correta do Python para o projeto
- Instalar todas as dependências necessárias
