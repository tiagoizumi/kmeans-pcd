import numpy as np
import csv

n=1000000           # Quantidade de pontos
k=16             # Quantidade de centróides
np.random.seed(2112)

# Gera os pontos e os centróides iniciais
dados = np.random.uniform(0, 1000, n)
centroides = np.random.uniform(0, 1000, k)

# Salva os dados em data.csv
with open("dados.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for x in dados:
        writer.writerow([x])

# Salva os centróides em centroides_iniciais.csv
with open("centroides_iniciais.csv", "w", newline="") as f:
    writer = csv.writer(f)
    for c in centroides:
        writer.writerow([c])

print(f" {n} pontos e {k} centróides")
