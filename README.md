🐆 Classificação de Comentários sobre Onças, Caseiro e Fake News usando BERT

Este projeto realiza a preparação de dados, treinamento de modelos BERT e avaliação automática para classificar comentários de notícias nas seguintes categorias:

Onça → sentimento sobre a onça (negativo, neutro, positivo)

Caseiro → sentimento sobre o caseiro atacado (negativo, neutro, positivo)

Fake News → identificação de desinformação (sim ou não)

Os comentários foram extraídos de notícias reais sobre ataques de onças no Brasil, e o objetivo é analisar automaticamente como o público reage em cada perspectiva.

📦 Funcionalidades do Projeto

🧹 Limpeza e preparação automática dos dados

🏷️ Geração de datasets separados por categoria

🤖 Treinamento de 3 modelos independentes usando BERT em português

📈 Gráfico de evolução do loss para cada modelo

📝 Avaliação completa com precisão, recall, F1-score

🔍 Geração de matriz de confusão

⚠️ Identificação de erros do modelo (exemplos mal classificados)

🧠 Tecnologias Utilizadas

Python

Pandas

Scikit-learn

PyTorch

Transformers (HuggingFace)

Matplotlib

Google Colab (recomendado para treinamento com GPU)

VS Code (organização e execução dos scripts)

📁 Estrutura do Projeto
noticias-de-oncas/
│
├── data_prep.py          # Prepara e gera os arquivos de treino/val/teste
├── train_bert.py         # Treina o modelo BERT para uma categoria
├── evaluate_bert.py      # Avalia o modelo treinado
├── oncas_comentarios.csv # Arquivo original de comentários
├── train_onca.csv
├── val_onca.csv
├── test_onca.csv
├── train_caseiro.csv
├── ...
│
└── requirements.txt

💻 Como Executar o Projeto (VS Code)
1. Instale os programas necessários

Python (3.10+)

Visual Studio Code com extensão Python instalada

2. Instale as dependências

No terminal do VS Code:

py -m pip install -r requirements.txt

3. Prepare o dataset

Coloque seu arquivo original na pasta:

oncas_comentarios.csv


Importante: o arquivo deve estar no formato CSV separado por ponto e vírgula (;).

Agora execute:

py data_prep.py


Isso irá gerar automaticamente:

train_onca.csv

val_onca.csv

test_onca.csv

train_caseiro.csv

val_caseiro.csv

test_caseiro.csv

train_fakenews.csv

val_fakenews.csv

test_fakenews.csv

🚀 Treinando um modelo

No computador (VS Code):

py train_bert.py onca


ou:

py train_bert.py caseiro
py train_bert.py fakenews


⚠ Atenção: o treinamento é MUITO pesado no PC
➡ Recomenda-se usar o Google Colab com GPU.

🔥 Treinando no Google Colab (recomendado)
1. Suba sua pasta no Colab
2. Na primeira célula instale as dependências:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers pandas scikit-learn matplotlib

3. No lugar do sys.argv padrão, ative manualmente a categoria:
sys.argv = ["train_bert.py", "caseiro"]


Depois execute normalmente o restante do código.

🧪 Avaliando o Modelo

Depois de treinado, rode:

py evaluate_bert.py caseiro


Isso irá mostrar:

Precisão

Recall

F1-score

Matriz de Confusão

Comentários classificados incorretamente

📊 Exemplo de Saída do Gráfico de Treinamento

O gráfico Evolução do Loss mostra:

linha azul → erro no treino

linha laranja → erro na validação

Se a validação sobe enquanto o treino cai, significa overfitting.

📦 Arquivo requirements.txt
pandas
torch
transformers
scikit-learn
matplotlib
numpy

🎓 Objetivo Geral do Projeto

O objetivo deste trabalho é:

Treinar modelos de linguagem (BERT em português) capazes de classificar automaticamente comentários de notícias em três perspectivas: sentimento sobre a onça, sentimento sobre o caseiro e verificação de fake news.

O projeto combina:

processamento de linguagem natural (NLP),

mineração de texto,

análise de sentimentos,

detecção de desinformação,

aprendizado profundo (Deep Learning).
