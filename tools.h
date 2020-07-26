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
    if(M <= 32 && N <= 32){
        printf("%s:\n", msg);
        for(int i=0; i<M; ++i){
            for(int j=0; j<N; ++j){
                cout << fixed << setprecision(3) << (float)mat[i*M + j] << " ";
            }
            printf("\n");
        }
    }
}

static void cpuGemm(int n, CTYPE alpha, const ATYPE *A, const BTYPE *B, CTYPE beta, CTYPE *C){
  if(n > 1024){ 
      return; 
  }
  int i, j, k;
  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;
      for (k = 0; k < n; ++k) {
        prod += (float)A[k * n + i] * (float)B[j * n + k];
      }
      C[j * n + i] = (CTYPE) (float)((float)alpha * (float)prod + (float)beta * (float)C[j * n + i]);
    }
  }
}

