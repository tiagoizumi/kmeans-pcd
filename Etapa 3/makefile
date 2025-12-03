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
NP = 4

# Alvo padrão: compilar
all: $(OUT)

$(OUT): $(SRC)
	$(MPICC) $(CFLAGS) $(SRC) -o $(OUT)

# Rodar com MPI
run: $(OUT)
	mpirun -np $(NP) ./$(OUT) $(DATA) $(CENTROIDS) $(MAX_IT) $(EPS)

# Limpar binários
clean:
	rm -f $(OUT)
