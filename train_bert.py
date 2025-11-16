"""
train_bert.py
Treina o modelo BERT para uma das categorias.
Uso:
    python train_bert.py onca
    python train_bert.py caseiro
    python train_bert.py fakenews
"""
"""
Primeira linha do Colab:
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install transformers pandas scikit-learn matplotlib
"""

# Importa bibliotecas necessárias
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW # otimizador recomendado para Transformers
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#VSCODE: coloque esse código no vscode ao invés do que está abaixo.
# Verifica o argumento da categoria
if len(sys.argv) != 2:
    print("ATENÇÃO! Uso correto: python train_bert.py [onca|caseiro|fakenews]")
    sys.exit(1)

"""
COLAB: utiliza esses códigos no Colab ao invés do código acima (muito lento no pc)
# Define manualmente o argumento da categoria (simulando chamada via terminal)
sys.argv = ["train_bert.py", "onca"]
#sys.argv = ["train_bert.py", "caseiro"]
#sys.argv = ["train_bert.py", "fakenews"]]
"""

# Captura a categoria escolhida
categoria = sys.argv[1]
print(f"\nTreinando modelo para a categoria: {categoria.upper()}")

# Classe personalizada para preparar o dataset para o BERT
class NewsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        # Lista de textos e rótulos
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

# Define número de classes (3 para onça/caseiro, 2 para fake news)
num_labels = 3 if categoria in ["onca", "caseiro"] else 2
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Usa GPU se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Carrega datasets de treino e validação
train_df = pd.read_csv(f"train_{categoria}.csv")
val_df = pd.read_csv(f"val_{categoria}.csv")

# Cria DataLoaders para treino e validação
train_loader = DataLoader(NewsDataset(train_df, tokenizer), batch_size=8, shuffle=True)
val_loader = DataLoader(NewsDataset(val_df, tokenizer), batch_size=8)

# Define otimizador e número de épocas
optimizer = AdamW(model.parameters(), lr=2e-5)
epochs = 10
train_losses, val_losses = [], []

# Loop de treinamento
for epoch in range(epochs):
    print(f"\nÉpoca {epoch + 1}/{epochs}")
    model.train()  # modo treino
    total_loss, total_val_loss = 0, 0
    preds, labels = [], []

    # Treino
    for batch in train_loader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}  # envia para GPU/CPU
        outputs = model(**batch)  # forward
        loss = outputs.loss
        loss.backward()  # backpropagation
        optimizer.step()  # atualização dos pesos
        total_loss += loss.item()

    # Validação
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            total_val_loss += outputs.loss.item()
            # Predições e rótulos para cálculo da acurácia
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())

    # Calcula acurácia da validação
    val_acc = accuracy_score(labels, preds)
    print(f"Loss treino: {total_loss/len(train_loader):.4f} | Loss val: {total_val_loss/len(val_loader):.4f} | Acurácia val: {val_acc:.3f}")

    # Armazena perdas para plotar depois
    train_losses.append(total_loss / len(train_loader))
    val_losses.append(total_val_loss / len(val_loader))

# Salva pesos do modelo treinado
torch.save(model.state_dict(), f"bert_{categoria}.pt")
print(f"\n Modelo salvo: bert_{categoria}.pt")

# Plota gráfico da evolução do loss
plt.plot(train_losses, label="Treino")
plt.plot(val_losses, label="Validação")
plt.legend()
plt.xlabel("Época")
plt.ylabel("Loss")
plt.title(f"Evolução do Loss ({categoria})")
plt.savefig(f"loss_{categoria}.png")
plt.show()

print("Modelo carregado com sucesso.") # teste