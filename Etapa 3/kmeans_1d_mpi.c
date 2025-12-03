#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include "mpi.h"

/* ---------- utilitários CSV ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    double *A = (double*)malloc((size_t)R * sizeof(double));
if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }
    FILE *f = fopen(path, "r");
if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok) continue;
        A[r++] = atof(tok);
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- Etapas do K-means ---------- */

/* Assignment: associa cada ponto X[i] ao centróide mais próximo */
static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;
    for(int i=0;i<N;i++){
        int best = -1;
        double bestd = 1e300;
        for(int c=0;c<K;c++){
            double diff = X[i] - C[c];
            double d = diff * diff;
            if(d < bestd){ bestd = d; best = c; }
        }
        assign[i] = best;
        sse += bestd;
    }
    return sse;
}

/* ASSIGNMENT LOCAL (MPI) - cada processo calcula o seu pedaço*/
double assignment_step_1d_local(const double *X_local,
                                const double *C,
                                int *assign_local,
                                int Nlocal, int K)
{
    double sse = 0.0;

    for(int i=0; i<Nlocal; i++){
        double bestd = 1e300;
        int best = -1;

        for(int c=0; c<K; c++){
            double diff = X_local[i] - C[c];
            double d = diff * diff;
            if(d < bestd){ bestd = d; best = c; }
        }

        assign_local[i] = best;
        sse += bestd;
    }
    return sse;
}


/* update: média dos pontos de cada cluster (1D)
se cluster vazio, copia X[0] (estratégia naive) */
static void update_step_1d(const double *X, double *C, const int *assign, int N, int K) {
    double *sum = (double*)calloc((size_t)K, sizeof(double));
    int *cnt = (int*)calloc((size_t)K, sizeof(int));
    if(!sum || !cnt){ fprintf(stderr,"Sem memoria no update\n"); exit(1); }
    for(int i=0;i<N;i++){
        int a = assign[i];
        cnt[a] += 1;
        sum[a] += X[i];
    }
    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else C[c] = X[0]; /* simples: cluster vazio recebe o primeiro */
    }
    free(sum); free(cnt);
}

/* UPDATE GLOBAL (MPI) usando Allreduce */
void update_step_1d_parallel(const double *X_local, const int *assign_local,
                             double *C, int Nlocal, int K, MPI_Comm comm)
{
    double *sum_local  = calloc(K, sizeof(double));
    int    *count_local = calloc(K, sizeof(int));

    for(int i=0; i<Nlocal; i++){
        int a = assign_local[i];
        sum_local[a]  += X_local[i];
        count_local[a] += 1;
    }

    double *sum_global  = calloc(K, sizeof(double));
    int    *count_global = calloc(K, sizeof(int));

    MPI_Allreduce(sum_local,  sum_global,  K, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(count_local,count_global,K, MPI_INT,    MPI_SUM, comm);

    for(int c=0; c<K; c++){
        if(count_global[c] > 0)
            C[c] = sum_global[c] / (double)count_global[c];
        /* senão mantém valor existente */
    }

    free(sum_local);  free(count_local);
    free(sum_global); free(count_global);
}


/* ---------- Loop principal ---------- */
static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        /* parada por variação relativa do SSE */
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

/* K-MEANS COMPLETO EM MPI */
void kmeans_1d_mpi(double *X, double *C, int *assign,
                   int N, int K, int max_iter, double eps,
                   MPI_Comm comm, int rank, int nprocs,
                   const char *outAssign, const char *outCentroid)
{
    /* ---------- Distribuição dos dados ---------- */
    int *N_local = malloc(nprocs * sizeof(int));
    int *offset  = malloc(nprocs * sizeof(int));

    int base = N / nprocs;
    int resto = N % nprocs;

    for(int i=0; i<nprocs; i++)
        N_local[i] = base + (i < resto);

    offset[0] = 0;
    for(int i=1; i<nprocs; i++)
        offset[i] = offset[i-1] + N_local[i-1];

    double *X_local = malloc(N_local[rank] * sizeof(double));
    int    *assign_local = malloc(N_local[rank] * sizeof(int));

    MPI_Scatterv(X, N_local, offset, MPI_DOUBLE,
                 X_local, N_local[rank], MPI_DOUBLE,
                 0, comm);

    /* ---------- Loop principal ---------- */
    double prev_sse = 1e300;

    for(int iter=0; iter<max_iter; iter++){

        /* PASSO 1: assignment local */
        double sse_local = assignment_step_1d_local(
            X_local, C, assign_local, N_local[rank], K);

        /* REDUÇÃO DO SSE */
        double sse_global;
        MPI_Allreduce(&sse_local, &sse_global, 1,
                      MPI_DOUBLE, MPI_SUM, comm);

        /* Critério de parada */
        double rel = fabs(sse_global - prev_sse) / prev_sse;
        prev_sse = sse_global;

        if(rel < eps) break;

        /* PASSO 2: update global */
        update_step_1d_parallel(
            X_local, assign_local, C,
            N_local[rank], K, comm);
    }

    /* ---------- Recolher assign total ---------- */
    MPI_Gatherv(assign_local, N_local[rank], MPI_INT,
                assign, N_local, offset, MPI_INT,
                0, comm);

    /* ---------- Rank 0 escreve arquivos ---------- */
    if(rank == 0){
        write_assign_csv(outAssign, assign, N);
        write_centroids_csv(outCentroid, C, K);
    }

    free(N_local); free(offset);
    free(X_local); free(assign_local);
}


/* ---------- main ---------- */
int main(int argc, char **argv){

    MPI_Init(&argc, &argv);

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if(argc < 3){
        if(rank == 0)
            printf("Uso: %s dados.csv centroides.csv [max_iter] [eps] assign.csv centroids.csv\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : "assign.csv";
    const char *outCentroid = (argc>6)? argv[6] : "centroids.csv";

    int N=0, K=0;
    double *X = NULL;
    double *C = NULL;
    int *assign = NULL;

    /* Apenas o rank 0 lê os arquivos */
    if(rank == 0){
        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign = malloc(N * sizeof(int));
    }

    /* Broadcast de N e K */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0){
        C = malloc(K * sizeof(double));
        assign = malloc(N * sizeof(int));
    }

    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Executar K-means MPI */
    kmeans_1d_mpi(X, C, assign,
                  N, K, max_iter, eps,
                  MPI_COMM_WORLD, rank, nprocs,
                  outAssign, outCentroid);

    MPI_Finalize();
    return 0;
}
