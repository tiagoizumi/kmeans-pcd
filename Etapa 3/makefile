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

# Rodar 20 vezes
run20: $(OUT)
	for i in $$(seq 1 20); do \
		echo ">>> Execução $$i com NP=$(NP)"; \
		mpirun -np $(NP) ./$(OUT) $(DATA) $(CENTROIDS) $(MAX_IT) $(EPS); \
	done

# Rodar 10 vezes csv
run20csv: $(OUT)
	echo "execucao,tempo_total_ms,allreduce_ms" > resultadosNP$$NP.csv
	for i in $$(seq 1 20); do \
		echo ">>> Execução $$i com NP=$(NP)"; \
		OUTPUT=$$(mpirun -np $(NP) ./$(OUT) $(DATA) $(CENTROIDS) $(MAX_IT) $(EPS)); \
		ALLR=$$(echo "$$OUTPUT" | grep -oP 'Tempo total de Allreduce:\s*\K[0-9.]+'); \
		TEMPO=$$(echo "$$OUTPUT" | grep -oP 'Tempo:\s*\K[0-9.]+'); \
		echo "$$i,$$TEMPO,$$ALLR" >> resultadosNP$$NP.csv; \
	done

# Limpar binários
clean:
	rm -f $(OUT)
