"""
data_prep.py
Lê a planilha e cria três versões do dataset:
1) Onça - (positivo = 2, neutro = 1, negativo = 0)
2) Caseiro - (positivo = 2, neutro = 1, negativo = 0)
3) Fake News - (sim = 1, não = 0)
"""

# Importa a biblioteca pandas para manipulação de dados em tabelas
import pandas as pd
# Importa função para dividir dataset em treino/validação/teste
from sklearn.model_selection import train_test_split

# Lê o arquivo original contendo os comentários
df = pd.read_csv("oncas_comentarios.csv", sep=";")

# Remove linhas sem texto e comentários duplicados
df = df.dropna(subset=["comment_text"]).drop_duplicates(subset=["comment_text"])

# Função auxiliar para preparar e salvar datasets balanceados
def preparar_dataset(coluna, mapa_rotulos, nome_saida):
    print(f"\nProcessando categoria: {coluna}")

    # Seleciona apenas a coluna de texto e a coluna de rótulo desejada
    temp = df[["comment_text", coluna]].dropna()
    # Renomeia a coluna de rótulo para "label"
    temp = temp.rename(columns={coluna: "label"})
    # Converte os rótulos textuais para valores numéricos conforme o mapa fornecido
    temp["label_id"] = temp["label"].map(mapa_rotulos)

    # Remove linhas com rótulos que não estão no mapa
    temp = temp.dropna(subset=["label_id"])

    # Divide em treino (70%), validação (15%) e teste (15%), estratificando pelos rótulos
    train_df, temp_df = train_test_split(temp, test_size=0.3, random_state=42, stratify=temp["label_id"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label_id"])

    # Salva os três datasets em arquivos CSV separados
    train_df.to_csv(f"train_{nome_saida}.csv", index=False)
    val_df.to_csv(f"val_{nome_saida}.csv", index=False)
    test_df.to_csv(f"test_{nome_saida}.csv", index=False)
    print(f"Arquivos salvos: train_{nome_saida}.csv, val_{nome_saida}.csv, test_{nome_saida}.csv")

# Cria datasets para cada categoria com seus respectivos mapeamentos
preparar_dataset("onca", {"negativo": 0, "neutro": 1, "positivo": 2}, "onca")
preparar_dataset("caseiro", {"negativo": 0, "neutro": 1, "positivo": 2}, "caseiro")
preparar_dataset("fake news", {"não": 0, "sim": 1}, "fakenews")

# Mensagem final de sucesso
print("\nTodos os datasets foram gerados com sucesso!")