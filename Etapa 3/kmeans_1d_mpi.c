/* kmeans_1d_mpi.c
   Versão MPI corrigida para Etapa 3 do Projeto PCD:
   - Quando cluster ficar vazio, copia X[0] (conforme enunciado).
   - Imprime iterações e SSE final (além do tempo).
   - Mantém distribuição com Scatterv/Gatherv e Allreduce para sums/counts.
   Compilar:
     mpicc -O2 -std=c99 kmeans_1d_mpi.c -o kmeans_1d_mpi -lm
   Executar:
     mpirun -np 4 ./kmeans_1d_mpi dados.csv centroides_iniciais.csv [max_iter] [eps] [assign.csv] [centroids.csv]
*/

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
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

/* ---------- Etapas do K-means ---------- */

/* Assignment: associa cada ponto X[i] ao centróide mais próximo (local) */
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

/* UPDATE GLOBAL (MPI) usando Allreduce
   Se count_global[c] == 0, define C[c] = X0 (conforme enunciado) */
void update_step_1d_parallel(const double *X_local, const int *assign_local,
                             double *C, int Nlocal, int K, MPI_Comm comm,
                             double X0)
{
    double *sum_local  = calloc((size_t)K, sizeof(double));
    int    *count_local = calloc((size_t)K, sizeof(int));
    if(!sum_local || !count_local){ fprintf(stderr,"Sem memoria no update_local\n"); MPI_Abort(comm,1); }

    for(int i=0; i<Nlocal; i++){
        int a = assign_local[i];
        if(a >= 0 && a < K){
            sum_local[a]  += X_local[i];
            count_local[a] += 1;
        }
    }

    double *sum_global  = calloc((size_t)K, sizeof(double));
    int    *count_global = calloc((size_t)K, sizeof(int));
    if(!sum_global || !count_global){ fprintf(stderr,"Sem memoria no update_global\n"); MPI_Abort(comm,1); }

    MPI_Allreduce(sum_local,  sum_global,  K, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(count_local,count_global,K, MPI_INT,    MPI_SUM, comm);

    for(int c=0; c<K; c++){
        if(count_global[c] > 0)
            C[c] = sum_global[c] / (double)count_global[c];
        else
            C[c] = X0; /* conforme enunciado: cluster vazio recebe X[0] */
    }

    free(sum_local);  free(count_local);
    free(sum_global); free(count_global);
}

/* ---------- K-means MPI completo ---------- */
void kmeans_1d_mpi(double *X, double *C, int *assign,
                   int N, int K, int max_iter, double eps,
                   MPI_Comm comm, int rank, int nprocs,
                   const char *outAssign, const char *outCentroid,
                   int *iters_out)
{
    /* ---------- Distribuição dos dados ---------- */
    int *N_local = malloc(nprocs * sizeof(int));
    int *offset  = malloc(nprocs * sizeof(int));
    if(!N_local || !offset){ if(rank==0) fprintf(stderr,"Sem memoria N_local/offset\n"); MPI_Abort(comm,1); }

    int base = N / nprocs;
    int resto = N % nprocs;

    for(int i=0; i<nprocs; i++)
        N_local[i] = base + (i < resto ? 1 : 0);

    offset[0] = 0;
    for(int i=1; i<nprocs; i++)
        offset[i] = offset[i-1] + N_local[i-1];

    double *X_local = malloc((size_t)N_local[rank] * sizeof(double));
    int    *assign_local = malloc((size_t)N_local[rank] * sizeof(int));
    if(!X_local || !assign_local){ fprintf(stderr,"Rank %d: sem memória para blocos locais\n", rank); MPI_Abort(comm,1); }

    MPI_Scatterv(X, N_local, offset, MPI_DOUBLE,
                 X_local, N_local[rank], MPI_DOUBLE,
                 0, comm);

    /* ---------- Precisamos do X[0] global (conforme enunciado) ---------- */
    double X0 = 0.0;
    if(rank == 0){
        X0 = X[0];
    }
    MPI_Bcast(&X0, 1, MPI_DOUBLE, 0, comm);

    /* ---------- Loop principal ---------- */
    double prev_sse = 1e300;
    double sse_global = 0.0;
    int iters = max_iter;

    for(int iter=0; iter<max_iter; iter++){

        /* PASSO 1: assignment local */
        double sse_local = assignment_step_1d_local(
            X_local, C, assign_local, N_local[rank], K);

        /* REDUÇÃO DO SSE (soma global) */
        MPI_Allreduce(&sse_local, &sse_global, 1,
                      MPI_DOUBLE, MPI_SUM, comm);

        /* Critério de parada (variação relativa do SSE) */
        double rel = fabs(sse_global - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        prev_sse = sse_global;

        /* Atualização global dos centróides */
        update_step_1d_parallel(
            X_local, assign_local, C,
            N_local[rank], K, comm, X0);

        if(rel < eps){
            iters = iter + 1; /* contabiliza a iteração atual */
            break;
        }
        /* se não convergiu, continua; se chegar ao fim do for, iters permanece max_iter */
        if(iter == max_iter - 1) iters = max_iter;
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
    
    *iters_out = iters;
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
    int iters = 0;
    double *X = NULL;
    double *C = NULL;
    int *assign = NULL;
    struct timeval start, end;

    /* Apenas o rank 0 lê os arquivos */
    if(rank == 0){
        X = read_csv_1col(pathX, &N);
        C = read_csv_1col(pathC, &K);
        assign = malloc((size_t)N * sizeof(int));
        if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); MPI_Abort(MPI_COMM_WORLD,1); }
        gettimeofday(&start, NULL);
    }

    /* Broadcast de N e K */
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&K, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0){
        C = malloc((size_t)K * sizeof(double));
        assign = malloc((size_t)N * sizeof(int));
        if(!C || !assign){ fprintf(stderr,"Rank %d: sem memoria para C/assign\n", rank); MPI_Abort(MPI_COMM_WORLD,1); }
    }

    /* Broadcast centróides iniciais para todos */
    MPI_Bcast(C, K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Executar K-means MPI */
    kmeans_1d_mpi(X, C, assign,
                  N, K, max_iter, eps,
                  MPI_COMM_WORLD, rank, nprocs,
                  outAssign, outCentroid,
                  &iters);


    MPI_Finalize();

    if(rank == 0) {
        gettimeofday(&end, NULL);
        double ms = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0;
        /* Recomputar SSE final */
        double final_sse = 0.0;
        for(int i=0;i<N;i++){
            int a = assign[i];
            double diff = X[i] - C[a];
            final_sse += diff * diff;
        }
        printf("K-means 1D (MPI)\n");
        printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
        printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, final_sse, ms);
    }

    if(X) free(X);
    if(C) free(C);
    if(assign) free(assign);

    return 0;
}
