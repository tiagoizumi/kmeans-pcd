# Nome dos arquivos
SRC = cpu_openmp.c
OUT = cpu_openmp
DATA = ../Dados/dados.csv
CENTROIDS = ../Dados/centroides_iniciais.csv
ASSIGN = assign.csv
FINAL_CENTROIDS = centroids.csv

# Compilador e flags
CC = gcc
CFLAGS = -O2 -fopenmp -std=c99 -lm

# Parâmetros de execução
MAX_ITER = 50
EPS = 1e-4

# Número de execuções
RUNS = 20

# Número de threads (default = 1)
THREADS ?= 1

# Alvo padrão
all: run

# Compilação
$(OUT): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) $(LDFLAGS)

# Execução
run: $(OUT)
	@echo "Executando com $(THREADS) thread(s)"
	@OMP_NUM_THREADS=$(THREADS) \
	./$(OUT) $(DATA) $(CENTROIDS) $(MAX_ITER) $(EPS) $(ASSIGN) $(FINAL_CENTROIDS)

# Execução 20 vezes
runn: $(OUT)
	@echo "Executando $(RUNS) vezes com $(THREADS) thread(s)"
	@for i in `seq 1 $(RUNS)`; do \
# 		echo ">> Execução $$i"; \
		OMP_NUM_THREADS=$(THREADS) \
		./$(OUT) $(DATA) $(CENTROIDS) $(MAX_ITER) $(EPS) $(ASSIGN) $(FINAL_CENTROIDS); \
	done
	@echo "Execuções finalizadas!"

# Limpeza
clean:
	rm -f $(OUT) $(ASSIGN) $(FINAL_CENTROIDS)

.PHONY: all run runn clean
