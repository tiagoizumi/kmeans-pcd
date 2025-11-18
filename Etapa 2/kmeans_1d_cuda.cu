
#define TIC(t) gettimeofday(&t, NULL)
#define TOC_MS(t0,t1) (((t1.tv_sec - t0.tv_sec)*1000.0) + ((t1.tv_usec - t0.tv_usec)/1000.0))

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <cuda.h>

/* ---------- utilitários CSV (float) ---------- */
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

static float *read_csv_1col_float(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    float *A = (float*)malloc((size_t)R * sizeof(float));
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
        char *tok = strtok(line, ",; \t");
        if(!tok) continue;
        A[r++] = (float)atof(tok);
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    FILE *f = fopen(path, "w");
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

static void write_centroids_csv(const char *path, const float *C, int K){
    FILE *f = fopen(path, "w");
    for(int i=0;i<K;i++) fprintf(f, "%.6f\n", C[i]);
    fclose(f);
}

/* ---------- GPU kernels ---------- */

// Cada thread escolhe o cluster mais próximo
__global__ void assign_clusters(const float *X, const float *C, int *assign, float *sse, int N, int K) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= N) return;

    float xi = X[index];

    float diff0 = xi - C[0];
    float best_dist = diff0 * diff0;
    int best_k = 0;

    for (int k = 1; k < K; k++) {
        float diff = xi - C[k];
        float dist = diff * diff;
        if (dist < best_dist) {
            best_dist = dist;
            best_k = k;
        }
    }

    assign[index] = best_k;

    // Acumula SSE
    atomicAdd(sse, best_dist);
}

__global__ void partial_reduce(const float *X, const int *assign,
                               float *sum, int *count,
                               int N, int K){
    extern __shared__ float shmem[];  // Memória compartilhada para K floats (sum) e K ints (count)


    float *local_sum = shmem;
    int   *local_count = (int*)( (char*)shmem + K*sizeof(float) );

    int tid = threadIdx.x;

    // Inicializa vetores locais de soma e contagem na memória compartilhada
    for (int k = tid; k < K; k += blockDim.x) {
        local_sum[k] = 0.0f;
        local_count[k] = 0;
    }
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < N){
        int k = assign[index];
        if(k >= 0 && k < K){
            atomicAdd(&local_sum[k], X[index]);
            atomicAdd(&local_count[k], 1);
        }
    }
    __syncthreads();

    for(int k = tid; k < K; k += blockDim.x){
        atomicAdd(&sum[k],   local_sum[k]);
        atomicAdd(&count[k], local_count[k]);
    }
}

/* ---------- MAIN ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides.csv [max_iter] [eps] [assign] [centroids] [blocksize]\n", argv[0]);
        return 1;
    }

    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    float eps   = (argc>4)? (float)atof(argv[4]) : 1e-4f;
    const char *outAssign   = (argc>5)? argv[5] : "assign.csv";
    const char *outCentroid = (argc>6)? argv[6] : "centroids.csv";
    int block = (argc > 7) ? atoi(argv[7]) : 256;

    float *h_X, *h_C;
    float *d_X, *d_C, *d_sum, *d_sse;
    int *d_assign, *d_count;
    int N=0, K=0;

    h_X = read_csv_1col_float(pathX, &N);
    h_C = read_csv_1col_float(pathC, &K);

    int *h_assign = (int*)malloc(N*sizeof(int));
    float *h_sum_host = (float*)malloc(K*sizeof(float));
    int *h_count_host = (int*)malloc(K*sizeof(int));
    float *newC = (float*)malloc(K*sizeof(float));

    cudaMalloc(&d_X, N*sizeof(float));
    cudaMalloc(&d_C, K*sizeof(float));
    cudaMalloc(&d_assign, N*sizeof(int));
    cudaMalloc(&d_sum, K*sizeof(float));
    cudaMalloc(&d_count, K*sizeof(int));
    cudaMalloc(&d_sse, sizeof(float));

    /* ---- MEDIDORES DE TEMPO ---- */
    struct timeval t0, t1;
    double t_h2d = 0.0, t_d2h = 0.0, t_kernel = 0.0;

    /* ---- H2D: dados iniciais ---- */
    gettimeofday(&t0, NULL);
    cudaMemcpy(d_X, h_X, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, K*sizeof(float), cudaMemcpyHostToDevice);
    gettimeofday(&t1, NULL);
    t_h2d += TOC_MS(t0, t1);

    int grid = (N + block - 1) / block;

    float prev_sse = -1.0f;
    float h_sse = 0.0f;

    printf("Grid=%d Block=%d\n", grid, block);

    struct timeval start_total, end_total;
    gettimeofday(&start_total, NULL);

    int it;
    for(it=0; it < max_iter; it++){

        cudaMemset(d_sse, 0, sizeof(float));

        /* ---- kernel: assign_clusters ---- */
        gettimeofday(&t0, NULL);
        assign_clusters<<<grid, block>>>(d_X, d_C, d_assign, d_sse, N, K);
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        t_kernel += TOC_MS(t0, t1);

        /* ---- D2H: SSE ---- */
        gettimeofday(&t0, NULL);
        cudaMemcpy(&h_sse, d_sse, sizeof(float), cudaMemcpyDeviceToHost);
        gettimeofday(&t1, NULL);
        t_d2h += TOC_MS(t0, t1);

        if (it == 0) prev_sse = h_sse;

        cudaMemset(d_sum, 0, K*sizeof(float));
        cudaMemset(d_count, 0, K*sizeof(int));
        cudaDeviceSynchronize();

        /* ---- kernel: partial_reduce ---- */
        size_t shmem = K*sizeof(float) + K*sizeof(int);
        gettimeofday(&t0, NULL);
        partial_reduce<<<grid, block, shmem>>>(d_X, d_assign, d_sum, d_count, N, K);
        cudaDeviceSynchronize();
        gettimeofday(&t1, NULL);
        t_kernel += TOC_MS(t0, t1);

        /* ---- D2H: reduce results ---- */
        gettimeofday(&t0, NULL);
        cudaMemcpy(h_sum_host,   d_sum,   K*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_count_host, d_count, K*sizeof(int),   cudaMemcpyDeviceToHost);
        gettimeofday(&t1, NULL);
        t_d2h += TOC_MS(t0, t1);

        float centroid_movement = 0.0f;
        for(int k=0;k<K;k++){
            newC[k] = (h_count_host[k] > 0 ? h_sum_host[k] / h_count_host[k] : h_X[0]);
            centroid_movement += fabsf(newC[k] - h_C[k]);
            h_C[k] = newC[k];
        }

        /* ---- H2D: enviar centróides ---- */
        gettimeofday(&t0, NULL);
        cudaMemcpy(d_C, h_C, K*sizeof(float), cudaMemcpyHostToDevice);
        gettimeofday(&t1, NULL);
        t_h2d += TOC_MS(t0, t1);

        if(it > 0 && prev_sse > 0.0f && h_sse > 0.0f){
            float rel = fabsf(prev_sse - h_sse) / prev_sse;
            if(rel < eps) { it++; break; }
        }

        if(h_sse > 0.0f)
            prev_sse = h_sse;

        printf("Iter %d SSE %.6f\n", it, h_sse);
    }

    /* ---- D2H final: assign ---- */
    gettimeofday(&t0, NULL);
    cudaMemcpy(h_assign, d_assign, N*sizeof(int), cudaMemcpyDeviceToHost);
    gettimeofday(&t1, NULL);
    t_d2h += TOC_MS(t0, t1);

    gettimeofday(&end_total, NULL);
    double tempo_total = TOC_MS(start_total, end_total);

    printf("\nK-means CUDA Finalizado\n");
    printf("Iterações: %d | Tempo total: %.2f ms\n", it, tempo_total);
    printf("Tempo kernel: %.2f ms\n", t_kernel);
    printf("Tempo H2D: %.2f ms\n", t_h2d);
    printf("Tempo D2H: %.2f ms\n", t_d2h);
    printf("SSE final: %.6f\n", h_sse);

    write_assign_csv(outAssign, h_assign, N);
    write_centroids_csv(outCentroid, h_C, K);

    cudaFree(d_X);
    cudaFree(d_C);
    cudaFree(d_assign);
    cudaFree(d_sum);
    cudaFree(d_count);
    cudaFree(d_sse);

    free(h_X);
    free(h_C);
    free(h_assign);
    free(h_sum_host);
    free(h_count_host);
    free(newC);

    return 0;
}
