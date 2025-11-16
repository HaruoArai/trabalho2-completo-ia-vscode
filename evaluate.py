"""
evaluate_bert.py
Avalia o modelo BERT para uma categoria específica.
Uso:
    python evaluate_bert.py onca
    python evaluate_bert.py caseiro
    python evaluate_bert.py fakenews
"""
# Importa bibliotecas necessárias
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix

#VSCODE: coloque esse código no vscode ao invés do que está abaixo.
# Argumento 
if len(sys.argv) != 2:
    print("Uso correto: python evaluate_bert.py [onca|caseiro|fakenews]")
    sys.exit(1)

categoria = sys.argv[1]
print(f"\nAvaliando modelo para a categoria: {categoria.upper()}")

"""
# Define manualmente a categoria que será avaliada
categoria = "onca"
#categoria = "caseiro"
#categoria = "fakenews"
print(f"\nAvaliando modelo para a categoria: {categoria.upper()}")
"""

# Classe personalizada para preparar o dataset para o BERT
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        # Extrai textos e rótulos do dataframe
        self.texts = df["comment_text"].tolist()
        self.labels = df["label_id"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        # Retorna o tamanho do dataset
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokeniza o texto no formato esperado pelo BERT
        encoding = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        # Retorna dicionário com input_ids, máscara de atenção e rótulo
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx])
        }

# Configurações do modelo
model_name = "neuralmind/bert-base-portuguese-cased"  # BERT em português
tokenizer = BertTokenizer.from_pretrained(model_name)
# Número de classes depende da categoria (3 para onça/caseiro, 2 para fake news)
num_labels = 3 if categoria in ["onca", "caseiro"] else 2

# Carrega modelo pré-treinado e pesos salvos
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.load_state_dict(torch.load(f"bert_{categoria}.pt", map_location="cpu"))
model.eval()  # coloca em modo avaliação

# Carrega dados de teste
test_df = pd.read_csv(f"test_{categoria}.csv")
test_loader = DataLoader(NewsDataset(test_df, tokenizer), batch_size=8)

# Avaliação do modelo
preds, labels = [], []
with torch.no_grad():  # desativa gradientes (não há treino)
    for batch in test_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = model(input_ids, attention_mask=attention_mask)
        # Obtém predições (classe com maior probabilidade)
        preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
        # Armazena rótulos verdadeiros
        labels.extend(batch["labels"].cpu().numpy())

# Relatório de desempenho
print("\nRelatório de Classificação:")
if categoria in ["onca", "caseiro"]:
    names = ["Negativo", "Neutro", "Positivo"]
else:
    names = ["Não", "Sim"]

# Exibe métricas de precisão, recall e f1-score
print(classification_report(labels, preds, target_names=names))
print("Matriz de confusão:")
print(confusion_matrix(labels, preds))

# Mostra exemplos de erros de classificação
print("\nExemplos de erros:")
for i in range(len(test_df)):
    if preds[i] != labels[i]:
        print(f"Texto: {test_df.iloc[i]['comment_text']}")
        print(f"Verdadeiro: {names[labels[i]]} | Previsto: {names[preds[i]]}\n")