#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h> 

double *read_csv(const char *path, int *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        printf("Erro ao abrir o arquivo %s\n", path);
        exit(1);
    }

    int linhas = 0;
    char linha[1000];

    while (fgets(linha, sizeof(linha), f))
        linhas++;

    double *valores = malloc(linhas * sizeof(double));
    if (!valores) {
        printf("Erro ao alocar memória\n");
        fclose(f);
        exit(1);
    }

    rewind(f);
    int i=0;
    while (fgets(linha, sizeof(linha), f)) {
        const char *delim = ",; \t";
        char *tok = strtok(linha, delim);
        if (tok)
            valores[i++] = atof(tok);
    }

    fclose(f);
    *n_out = linhas;
    return valores;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const double *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    for(int c=0;c<K;c++) fprintf(f, "%.6f\n", C[c]);
    fclose(f);
}

static double assignment_step_1d(const double *X, const double *C, int *assign, int N, int K){
    double sse = 0.0;

    #pragma omp parallel for reduction(+:sse) schedule(static)
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

static void update_step_1d(const double *X, double *C, const int *assign, int N, int K){
    int nthreads = omp_get_max_threads();

    double **sum_thread = (double**)malloc(nthreads * sizeof(double*));
    int **cnt_thread = (int**)malloc(nthreads * sizeof(int*));

    for(int t=0;t<nthreads;t++){
        sum_thread[t] = (double*)calloc(K, sizeof(double));
        cnt_thread[t] = (int*)calloc(K, sizeof(int));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(static)
        for(int i=0;i<N;i++){
            int a = assign[i];
            cnt_thread[tid][a] += 1;
            sum_thread[tid][a] += X[i];
        }
    }

    double *sum = (double*)calloc(K, sizeof(double));
    int *cnt = (int*)calloc(K, sizeof(int));

    for(int t=0;t<nthreads;t++){
        for(int c=0;c<K;c++){
            sum[c] += sum_thread[t][c];
            cnt[c] += cnt_thread[t][c];
        }
        free(sum_thread[t]);
        free(cnt_thread[t]);
    }
    free(sum_thread);
    free(cnt_thread);

    for(int c=0;c<K;c++){
        if(cnt[c] > 0) C[c] = sum[c] / (double)cnt[c];
        else           C[c] = X[0]; // cluster vazio => estratégia simples
    }

    free(sum);
    free(cnt);
}

static void kmeans_1d(const double *X, double *C, int *assign,
                      int N, int K, int max_iter, double eps,
                      int *iters_out, double *sse_out)
{
    double prev_sse = 1e300;
    double sse = 0.0;
    int it;
    for(it=0; it<max_iter; it++){
        sse = assignment_step_1d(X, C, assign, N, K);
        double rel = fabs(sse - prev_sse) / (prev_sse > 0.0 ? prev_sse : 1.0);
        if(rel < eps){ it++; break; }
        update_step_1d(X, C, assign, N, K);
        prev_sse = sse;
    }
    *iters_out = it;
    *sse_out = sse;
}

int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] assign.csv centroids.csv\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    double eps   = (argc>4)? atof(argv[4]) : 1e-4;
    const char *outAssign   = (argc>5)? argv[5] : "assign.csv";
    const char *outCentroid = (argc>6)? argv[6] : "centroids.csv";

    int N=0, K=0;
    double *X = read_csv(pathX, &N);
    double *C = read_csv(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));

    struct timeval start, end;
    gettimeofday(&start, NULL);

    int iters = 0; double sse = 0.0;
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);

    gettimeofday(&end, NULL);
    double ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                (end.tv_usec - start.tv_usec) / 1000.0;

    printf("K-means 1D (OpenMP)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, eps);
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.3f ms | Threads: %d\n",
           iters, sse, ms, omp_get_max_threads());

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}
