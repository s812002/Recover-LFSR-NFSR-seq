#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

// ==========================================
// 参数配置 (Joint LFSR+NFSR State)
// ==========================================
#define N              80
#define NUM_VARS       (2 * N)     // 160: [0..79]=NFSR(y), [80..159]=LFSR(x)
#define NUM_BITS       29500      // 矩阵行数
#define VERIFY_COUNT   2000
#define COEFF_COUNT    ( NUM_VARS + (NUM_VARS * (NUM_VARS - 1)) / 2) // 12881
#define ROWS           (NUM_BITS - NUM_VARS - VERIFY_COUNT) 

#define BITS_PER_WORD  64
#define TOTAL_COLS     (COEFF_COUNT + 1)
#define WORDS_PER_ROW  ((TOTAL_COLS + BITS_PER_WORD - 1) / BITS_PER_WORD)

#define STATE_WORDS    ((NUM_VARS + BITS_PER_WORD - 1) / BITS_PER_WORD)
static uint64_t *joint_packed = NULL;
static uint8_t target[NUM_BITS]; //Ax=b,b这一列


static const char *INPUT_FILE = "D:\\seq.txt";//文件路径

static void print_target_binary(const uint8_t *arr, size_t len) {
    printf("[*] Target sequence (%zu bits):\n", len);
    for (size_t i = 0; i < len; i++) {
        putchar(arr[i] ? '1' : '0');
        if ((i + 1) % 80 == 0) putchar('\n');
    }
    if (len % 80 != 0) putchar('\n');
}
//打印解的函数
static void print_solution(const uint8_t *sol) {
    printf("\n[*] === RECOVERED 160-VAR BOOLEAN FUNCTION ANF ===\n");
    printf("Recovered ANF terms: ");
    
    int idx = 0; 
    int term_count = 0;

    for (int i = 0; i < NUM_VARS; i++, idx++) {
        if (sol[idx]) {
            printf("+s%d ", i);
            term_count++;
        }
    }

    for (int i = 0; i < NUM_VARS; i++) {
        for (int j = i + 1; j < NUM_VARS; j++, idx++) {
            if (sol[idx]) {
                printf("+s%d*s%d ", i, j);
                term_count++;
            }
        }
    }
    if (term_count == 0) printf("(ZERO FUNCTION)");
    printf("\n");
    printf("[*] Total non-zero terms: %d / %d\n", term_count, COEFF_COUNT);
    printf("[*] ============================================\n\n");
}
//从文件中读取序列
static int generate_joint_sequence_from_file(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "[-] Cannot open input file: %s\n", filename);
        return -1;
    }

    size_t jp_size = (size_t)NUM_BITS * STATE_WORDS * sizeof(uint64_t);
    joint_packed = (uint64_t *)malloc(jp_size);
    if (!joint_packed) { 
        fprintf(stderr, "[-] Joint state OOM (%zu MB)\n", jp_size >> 20); 
        fclose(fp); 
        return -1; 
    }
    memset(joint_packed, 0, jp_size);
    memset(target, 0, sizeof(target));

    printf("[*] Reading sliding-window samples from '%s'...\n", filename);
    clock_t t0 = clock();

    // 1. 先将前 NUM_VARS (160) 位读入第一个样本的 joint_packed[0]
    int ch;
    int init_bits = 0;
    while (init_bits < NUM_VARS && (ch = fgetc(fp)) != EOF) {
        if (ch != '0' && ch != '1') continue;
        uint64_t *row = joint_packed; // 第0个样本
        if (ch == '1')
            row[init_bits / 64] |= (1ULL << (init_bits % 64));
        init_bits++;
    }

    if (init_bits < NUM_VARS) {
        fprintf(stderr, "[-] File too small to fill even one initial state (%d/%d bits)\n", 
                init_bits, NUM_VARS);
        free(joint_packed); joint_packed = NULL;
        fclose(fp);
        return -1;
    }

    // 2. 滑动窗口读取：每读1个新bit，生成1个新样本
    int sample_idx = 0;
    int total_bits_read = init_bits;

    while (sample_idx < NUM_BITS && (ch = fgetc(fp)) != EOF) {
        if (ch != '0' && ch != '1') continue; // 跳过非01字符
        total_bits_read++;

        uint8_t new_bit = (ch == '1') ? 1 : 0;

        target[sample_idx] = new_bit; 

        // --- 滑动生成下一个样本 (sample_idx + 1) ---
        if (sample_idx + 1 < NUM_BITS) {
            const uint64_t *cur_row  = joint_packed + (size_t)sample_idx * STATE_WORDS;
            uint64_t       *next_row = joint_packed + (size_t)(sample_idx + 1) * STATE_WORDS;

            // 整体右移1位
            // next[i] = cur[i+1] for i=0..158; next[159] = new_bit
            for (int i = 0; i < NUM_VARS - 1; i++) {
                int src_word = (i + 1) / 64;
                int src_bit  = (i + 1) % 64;
                uint8_t val = (cur_row[src_word] >> src_bit) & 1;
                
                int dst_word = i / 64;
                int dst_bit  = i % 64;
                if (val) next_row[dst_word] |= (1ULL << dst_bit);
            }
            // 最高位写入新读入的bit
            if (new_bit)
                next_row[(NUM_VARS - 1) / 64] |= (1ULL << ((NUM_VARS - 1) % 64));
        }

        sample_idx++;
    }

    fclose(fp);

    if (sample_idx < NUM_BITS) {
        fprintf(stderr, "[-] WARNING: Only read %d/%d sliding samples! (Need %d more valid bits)\n", 
                sample_idx, NUM_BITS, NUM_BITS - sample_idx);
    }

    printf("[*] Read %d sliding-window samples (%d total bits consumed) in %.2f s\n", 
           sample_idx, total_bits_read, (double)(clock() - t0) / CLOCKS_PER_SEC);
    return 0;
}

// ==========================================
//   构建矩阵
// ==========================================
static void build_matrix(uint64_t *matrix) {
    printf("[*] Building %d ROWs matrix (%.1f MB)...\n", 
           ROWS, (double)ROWS * WORDS_PER_ROW * 8 / 1e6);
    clock_t t0 = clock();
    memset(matrix, 0, (size_t)ROWS * WORDS_PER_ROW * sizeof(uint64_t));

    for (int t = 0; t < ROWS; t++) {
        uint64_t *mrow = matrix + (size_t)t * WORDS_PER_ROW;
        const uint64_t *srow = joint_packed + (size_t)t * STATE_WORDS;
        
        int col = 0;
        for (int i = 0; i < NUM_VARS; i++, col++) {
            if (srow[i / 64] & (1ULL << (i % 64)))
                mrow[col / 64] |= (1ULL << (col % 64));
        }
        
        // Quadratic: pairwise AND of joint state bits
        for (int i = 0; i < NUM_VARS; i++) {
            uint64_t si = (srow[i / 64] >> (i % 64)) & 1;
            if (!si) { col += (NUM_VARS - 1 - i); continue; }
            for (int j = i + 1; j < NUM_VARS; j++, col++) {
                if (srow[j / 64] & (1ULL << (j % 64)))
                    mrow[col / 64] |= (1ULL << (col % 64));
            }
        }
        
        
        if (target[t]) {
            int rc = TOTAL_COLS - 1;
            mrow[rc / 64] |= (1ULL << (rc % 64));
        }
    }
    printf("[*] Matrix built in %.2f s\n", (double)(clock() - t0) / CLOCKS_PER_SEC);
}
// ==========================================
//  GF(2) 上的高斯消元
// ==========================================
static int gauss_eliminate(uint64_t *matrix, uint8_t *solution) {
    printf("[*] Eliminating %d x %d system...\n", ROWS, COEFF_COUNT);
    clock_t t0 = clock();
    
    int *pivot_row = (int *)malloc(COEFF_COUNT * sizeof(int));
    if (!pivot_row) { fprintf(stderr, "[-] OOM\n"); return -1; }
    memset(pivot_row, -1, COEFF_COUNT * sizeof(int));
    
    int rank = 0;
    
   
    for (int col = 0; col < COEFF_COUNT && rank < ROWS; col++) {
        int wi = col / 64;
        uint64_t mask = 1ULL << (col % 64);
        int found = -1;
        for (int r = rank; r < ROWS; r++)
            if (matrix[(size_t)r * WORDS_PER_ROW + wi] & mask) { found = r; break; }
        if (found == -1) continue;
        
        if (found != rank) {
            uint64_t *a = matrix + (size_t)rank * WORDS_PER_ROW;
            uint64_t *b = matrix + (size_t)found * WORDS_PER_ROW;
            for (int w = 0; w < WORDS_PER_ROW; w++) { uint64_t t = a[w]; a[w] = b[w]; b[w] = t; }
        }
        pivot_row[col] = rank;
        uint64_t *piv = matrix + (size_t)rank * WORDS_PER_ROW;
        for (int r = 0; r < ROWS; r++) {
            if (r != rank) {
                uint64_t *cur = matrix + (size_t)r * WORDS_PER_ROW;
                if (cur[wi] & mask)
                    for (int w = wi; w < WORDS_PER_ROW; w++) cur[w] ^= piv[w];
            }
        }
        
        if (col % 2000 == 0) printf("    Progress: %d/%d (rank=%d)\n", col, COEFF_COUNT, rank);
        rank++;
    }
    
    printf("[*] Rank: %d/%d (%.2f s)\n", rank, COEFF_COUNT, (double)(clock()-t0)/CLOCKS_PER_SEC);
    
    memset(solution, 0, COEFF_COUNT);
    int bw = (TOTAL_COLS - 1) / 64;
    uint64_t bm = 1ULL << ((TOTAL_COLS - 1) % 64);

    for (int c = 0; c < COEFF_COUNT; c++) {
        if (pivot_row[c] == -1) {
            solution[c] = 1; // 自由变量赋1
        } else {
            uint64_t *pr = matrix + (size_t)pivot_row[c] * WORDS_PER_ROW;
            uint8_t val = (pr[bw] & bm) ? 1 : 0;
            for (int fc = 0; fc < COEFF_COUNT; fc++) {
                if (pivot_row[fc] == -1) {
                    int fwi = fc / 64;
                    uint64_t fmask = 1ULL << (fc % 64);
                    if (pr[fwi] & fmask) val ^= 1;
                }
            }
            solution[c] = val;
        }
    }
    
    int nz = 0; for (int i = 0; i < COEFF_COUNT; i++) if (solution[i]) nz++;
    printf("[DIAG] Non-zero coeffs: %d, Free vars: %d\n", 
           nz, COEFF_COUNT - rank);
    
    free(pivot_row);
    return 0;
}
// ==========================================
// 拿2000个bit验证
// ==========================================
static void verify(const uint8_t *sol) {
    printf("\n[*] === VERIFICATION (%d samples, out-of-training) ===\n", VERIFY_COUNT);
    int errors = 0;
    int start = ROWS;
    
    for (int k = 0; k < VERIFY_COUNT && (start + k) < NUM_BITS; k++) {
        int t = start + k;
        const uint64_t *srow = joint_packed + (size_t)t * STATE_WORDS;
        uint8_t pred = 0;
        int idx = 0; 
        for (int i = 0; i < NUM_VARS; i++, idx++)
            if (sol[idx] && (srow[i/64] & (1ULL<<(i%64)))) pred ^= 1;
        for (int i = 0; i < NUM_VARS; i++) {
            uint64_t si = (srow[i/64] >> (i%64)) & 1;
            if (!si) { idx += (NUM_VARS-1-i); continue; }
            for (int j = i+1; j < NUM_VARS; j++, idx++)
                if (sol[idx] && (srow[j/64] & (1ULL<<(j%64)))) pred ^= 1;
        }
        if (pred != target[t]) errors++;
    }
    
    printf("    Errors: %d / %d (%.2f%%)\n", errors, VERIFY_COUNT, 100.0*errors/VERIFY_COUNT);
    printf(errors == 0 ? "    SUCCESS\n" : "    FAILURE\n");
    printf("[*] === END ===\n\n");
}

// ==========================================
// Main
// ==========================================
int main(int argc, char *argv[]) {
    clock_t total = clock();
    
    const char *infile = INPUT_FILE;
    if (argc > 1) infile = argv[1];
    
    if (generate_joint_sequence_from_file(infile) != 0) {
        return 1;
    }
    
    size_t mat_bytes = (size_t)ROWS * WORDS_PER_ROW * sizeof(uint64_t);
    uint64_t *matrix = (uint64_t *)malloc(mat_bytes);
    uint8_t *solution = (uint8_t *)calloc(COEFF_COUNT, sizeof(uint8_t));
    if (!matrix || !solution) { fprintf(stderr, "[-] OOM\n"); return 1; }
    
    build_matrix(matrix);
    gauss_eliminate(matrix, solution);
    verify(solution);
    print_solution(solution); 
    printf("[*] Total: %.2f s\n", (double)(clock()-total)/CLOCKS_PER_SEC);
    print_target_binary(target, 160);
    
    free(joint_packed);
    free(matrix);
    free(solution);
    
    return 0;
}