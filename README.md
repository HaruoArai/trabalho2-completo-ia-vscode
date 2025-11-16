# 🐆 Notícias de onças classificadas usando BERT 

Este projeto utiliza BERT em português para realizar análise de sentimentos e detecção de fake news em comentários coletados no Youtube sobre um caso envolvendo um ataque de onça aqui na região de Mato Grosso do Sul e sua repercussão nas redes sociais.

O sistema classifica cada comentário em três tarefas distintas:
- Sentimento relacionado à Onça (positivo, neutro, negativo)
- Sentimento relacionado ao Caseiro (positivo, neutro, negativo)
- Detecção de Fake News (sim / não)

O projeto inclui pré-processamento, treinamento e avaliação de modelos independentes para cada categoria, utilizando o modelo pré-treinado neuralmind/bert-base-portuguese-cased.

## 📦 Funcionalidades

- Processamento de dataset e criação automática de arquivos: train_onca.csv, val_onca.csv, test_onca.csv train_caseiro.csv, val_caseiro.csv, test_caseiro.csv train_fakenews.csv, val_fakenews.csv e test_fakenews.csv
- Treinamento de 3 modelos BERT independentes
- Geração de gráficos de evolução de loss
- Cálculo de métricas de desempenho: Precisão, Recall, F1-score e Matriz de Confusão
- Exibição automática de erros de classificação para análise qualitativa

## 🧰 Tecnologias Utilizadas

- **Visual Studio Code (VS Code)**: ambiente de desenvolvimento recomendado e utilizado neste trabalho.
- **Google Colab**: plataforma online para execução de notebooks Python em nuvem, facilitando testes e compartilhamento.
- **Python**: linguagem principal do projeto.
- **Transformers (HuggingFace)**: biblioteca voltada para modelos de NLP (Processamento de Linguagem Natural), como BERT e GPT, usada para treinar e aplicar modelos de inteligência artificial.
- **Pandas**: manipulação e exportação de dados em formato tabular.
- **Scikit-learn**: biblioteca de machine learning para pré-processamento, treinamento, avaliação e métricas de modelos.
- **Matplotlib**: biblioteca de visualização de dados, utilizada para gerar gráficos e análises visuais.
- **CSV**: formato de saída dos dados coletados.

## 📁 Estrutura do Projeto
```bash
/noticia-de-oncas
│
├── data_prep.py
├── train_bert.py
├── evaluate_bert.py
│
├── train_onca.csv
├── val_onca.csv
├── test_onca.csv
│
├── train_caseiro.csv
├── val_caseiro.csv
├── test_caseiro.csv
│
├── train_fakenews.csv
├── val_fakenews.csv
├── test_fakenews.csv
│
├── bert_onca.pt
├── bert_caseiro.pt
├── bert_fakenews.pt
│
├── loss_onca.png
├── loss_caseiro.png
├── loss_fakenews.png
│
└── requirements.txt
```

## 💻 Como executar o projeto (Windows 10/11 + VS Code)

### 1. Instale os programas necessários

- [Visual Studio Code](https://code.visualstudio.com/) com a extensão **Python**
- [Python](https://www.python.org/downloads) (versão 3.10 ou superior)

---

### 2. Prepare o ambiente no VS Code

- Crie uma pasta chamada `noticias-de-oncas`
- Coloque os arquivos disponibilizados neste repositório dentro da pasta

Abra o terminal do VS Code (`Ctrl + Shift + '` ou vá em **Terminal → Novo Terminal**) e siga os passos abaixo:

#### a) Verifique se o Python está instalado

```bash
py --version
```

#### b) Instale as bibliotecas necessárias através do requeriments.txt

```bash
pip install -r requirements.txt
```

#### c) Prepare os datasets

```bash
py data_prep.py
```
Isso irá gerar automaticamente os arquivos de treino, validação e teste para cada categoria.

#### d) Treine um modelo

```bash
py train_bert.py onca
py train_bert.py caseiro
py train_bert.py fakenews
```
Ao final, será gerado um arquivo de pesos:
```bash
bert_onca.pt
bert_caseiro.pt
bert_fakenews.pt
```
E um gráfico:
```bash
loss_onca.png
loss_caseiro.png
loss_fakenews.png
```

#### e) Avalie o modelo

```bash
py evaluate_bert.py onca
py evaluate_bert.py caseiro
py evaluate_bert.py fakenews
```
A saída inclui:
-Métricas gerais
-Matriz de confusão
-Exemplos onde o modelo errou

---

## 📊 Sobre a Interpretação dos Sentimentos

Categoria: Onça
- Positivo → comentários que defendem a onça
- Neutro → comentários imparciais
- Negativo → comentários que culpam a onça

Categoria: Caseiro
- Positivo → comentários que defendem o caseiro
- Neutro → comentários imparciais
- Negativo → comentários que culpam o caseiro pelo ataque

Categoria: Fake News
- Sim → o comentário contém desinformação
- Não → não contém desinformação

Os modelos foram treinados com base nesses critérios exatamente como definidos no dataset original.

