#pragma once
using namespace std;

// CUBLAS ALGORITHMS
#define NUM_CUBLAS_ALGS 2
cublasGemmAlgo_t cublasAlgs[NUM_CUBLAS_ALGS] = {CUBLAS_GEMM_DEFAULT, 
                           CUBLAS_GEMM_DEFAULT_TENSOR_OP};


// CUBLAS MATH MODES
#define NUM_CUBLAS_MATH_MODES 4
cublasMath_t cublasMathModes[NUM_CUBLAS_MATH_MODES] = {CUBLAS_DEFAULT_MATH, 
                           CUBLAS_PEDANTIC_MATH,
                           CUBLAS_TF32_TENSOR_OP_MATH,
                           CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION};


// CUBLAS COMPUTE TYPES (for cublasGemmEx and cublasLtMatmul)
#define NUM_CUBLAS_COMPUTE_TYPES 11
cublasComputeType_t cublasComputeTypes[NUM_CUBLAS_COMPUTE_TYPES] = {CUBLAS_COMPUTE_16F,
                                                CUBLAS_COMPUTE_16F_PEDANTIC,
                                                CUBLAS_COMPUTE_32F,
                                                CUBLAS_COMPUTE_32F_PEDANTIC,
                                                CUBLAS_COMPUTE_32F_FAST_16F,
                                                CUBLAS_COMPUTE_32F_FAST_16BF,
                                                CUBLAS_COMPUTE_32F_FAST_TF32,
                                                CUBLAS_COMPUTE_64F,
                                                CUBLAS_COMPUTE_64F_PEDANTIC,
                                                CUBLAS_COMPUTE_32I,
                                                CUBLAS_COMPUTE_32I_PEDANTIC};
                                                 

// STRINGS FOR MESSAGES
// CUBLAS ALGORITHMS STRINGS
const char* cublasAlgsStr[NUM_CUBLAS_ALGS] = {"CUBLAS_GEMM_DEFAULT", 
                                              "CUBLAS_GEMM_DEFAULT_TENSOR_OP"};

// CUBLAS MATH MODES STRINGS
const char* cublasMathModesStr[NUM_CUBLAS_MATH_MODES] = {"CUBLAS_DEFAULT_MATH", 
                                   "CUBLAS_PEDANTIC_MATH",
                                   "CUBLAS_TF32_TENSOR_OP_MATH",
                                   "CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION"};


// CUBLAS COMPUTE TYPES STRINGS (for cublasGemmEx and cublasLtMatmul)
const char* cublasComputeTypesStr[NUM_CUBLAS_COMPUTE_TYPES] = {"CUBLAS_COMPUTE_16F",
                                        "CUBLAS_COMPUTE_16F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32F",
                                        "CUBLAS_COMPUTE_32F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32F_FAST_16F",
                                        "CUBLAS_COMPUTE_32F_FAST_16BF",
                                        "CUBLAS_COMPUTE_32F_FAST_TF32",
                                        "CUBLAS_COMPUTE_64F",
                                        "CUBLAS_COMPUTE_64F_PEDANTIC",
                                        "CUBLAS_COMPUTE_32I",
                                        "CUBLAS_COMPUTE_32I_PEDANTIC"};

void printDefines(const char** opts, int n, const char *msg){
    printf("%s:\n", msg);
    for(int i=0; i<4; ++i){
       cout << i << " = " << cublasMathModesStr[i] << endl;  
    }
}

void printArgsInfo(){
    printDefines(cublasMathModesStr, NUM_CUBLAS_MATH_MODES, "CUBLAS Math Modes");
}

template <typename T>
void print_matrix(T *mat, int M, int N, const char *msg){
    if(M <= LIM_PRINT_N && N <= LIM_PRINT_N){
        printf("%s:\n", msg);
        for(int i=0; i<M; ++i){
            for(int j=0; j<N; ++j){
                printf("%6.3f ", (float)mat[i*M + j]);
            }
            printf("\n");
        }
    }
}

static void cpuGemm(int n, CTYPE alpha, const ATYPE *A, const BTYPE *B, CTYPE beta, CTYPE *C){
    if(n > LIM_CHECK_N){ 
      return; 
    }
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float prod = 0;
            for (int k = 0; k < n; ++k) {
                prod += (float)A[k * n + i] * (float)B[j * n + k];
            }
            C[j * n + i] = (CTYPE) (float)((float)alpha * (float)prod + (float)beta * (float)C[j * n + i]);
        }
    }
}

template <typename T>
void fillMatrixRand(T *m, unsigned long nelem){
    #pragma omp parallel
    {
        random_device rd;
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<> dis(0.0, 1.0);
        long seg = (nelem+nt-1)/nt;
        long start = tid*seg;
        long end = start + seg;
        for(unsigned long i = start; i < nelem && i < end; i++){
            m[i] = (T)dis(gen);
        }
    }
}

void checkResult(CTYPE *h_C_ref, CTYPE *h_C, int N, unsigned long nelem, float *en, float *rn){
    float error_norm = 0;
    float ref_norm = 0;
    float max_diff = -1.0f;
    float diff;
    unsigned long maxindex = 0;
    if(N <= LIM_CHECK_N){
        for(int i = 0; i < nelem; ++i){
            diff = (float)h_C_ref[i] - (float)h_C[i];
            if(fabs(diff) > max_diff){
                max_diff = fabs(diff);
                maxindex = i;
            }
            error_norm += diff * diff;
            ref_norm += (float)h_C_ref[i] * (float)h_C_ref[i];
        }

        error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
        ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));
        if (fabs(ref_norm) < 1e-7) {
            fprintf(stderr, "!!!! reference norm is 0\n");
            return;
        }
        double errPerc = fabs(max_diff/(h_C_ref[maxindex]));
        //printf("MaxDiff %f        (C[%i] CPU: %f      GPU: %f)\n", max_diff, maxindex, h_C_ref[maxindex], h_C[maxindex]);
        printf("MaxErr %f%\n", errPerc);
    }
    else{
        printf("Skipping Check (matrix too large)\n"); fflush(stdout);
    }
    *en = error_norm;
    *rn = ref_norm;
}
