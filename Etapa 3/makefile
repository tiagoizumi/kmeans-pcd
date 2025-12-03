# Compilador e flags
MPICC = mpicc
CFLAGS = -O2 -std=c99 -lm

# Arquivos
SRC = kmeans_1d_mpi.c
OUT = kmeans_1d_mpi

DATA = dados.csv
CENTROIDS = centroides_iniciais.csv
MAX_IT = 50
EPS = 1e-4

# NP é parâmetro externo, padrão = 4
NP ?= 4

# Alvo padrão: compilar
all: $(OUT)

$(OUT): $(SRC)
	$(MPICC) $(CFLAGS) $(SRC) -o $(OUT)

# Rodar 20 vezes (sem salvar logs)
run20: $(OUT)
	for i in $$(seq 1 20); do \
		echo ">>> Execução $$i com NP=$(NP)"; \
		mpirun -np $(NP) ./$(OUT) $(DATA) $(CENTROIDS) $(MAX_IT) $(EPS); \
	done

# Rodar 20 vezes salvando logs
run20log: $(OUT)
	mkdir -p logs
	for i in $$(seq 1 20); do \
		echo ">>> Execução $$i com NP=$(NP)"; \
		mpirun -np $(NP) ./$(OUT) $(DATA) $(CENTROIDS) $(MAX_IT) $(EPS) > logs/run_$$i.txt; \
	done

# Limpar binários
clean:
	rm -f $(OUT)
