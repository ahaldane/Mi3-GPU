//Copyright 2019 Allan Haldane.
//
//This file is part of Mi3-GPU.
//
//Mi3-GPU is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, version 3 of the License.
//
//Mi3-GPU is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with Mi3-GPU.  If not, see <http://www.gnu.org/licenses/>.
//
//Contact: allan.haldane _AT_ gmail.com
//
//
// Compile me with:
// $ gcc -O3 mcmcCPUgenThreaded.c -lm -o cpu
//
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Random123/threefry.h"
#include "math.h"
#define NUM_THREADS 1
#if NUM_THREADS > 1
#include <pthread.h>
#endif

unsigned int q, L;
float *couplings;
#define coupling(i,j,a,b) couplings[(L*i + j)*q*q + ((int)a)*q + ((int)b)]
char *alphabet;
char *sseq;
uint loops, steps;

unsigned long int getRandSeed(){
    unsigned long int seed=0;
    FILE *f = fopen("/dev/urandom", "rb");
    if(fread(&seed, sizeof(seed), 1, f) != 1){
        fprintf(stderr, "Error: Could not read /dev/urandom to get a random "
                        "seed");
        exit(1);
    }
    fclose(f);
    return seed;
}

float uniformMap32(uint32_t i){
    //converts a 32 bit integer to a uniform float [0,1)
    return (i>>8)*0x1.0p-24f;
}

float getEnergy(char *seq){
    int i, j;
    float energy = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            energy += coupling(i,j,seq[i],seq[j]);
        }
    }
    return energy;
}

inline void
step(float *energy, char *seq, int pos, int res, float rng) {
    int m;

    float newenergy = *energy;
    uint sbn = seq[pos];

    for (m = 0; m < L; m++) {
        newenergy += (coupling(pos,m,res,seq[m]) -
                      coupling(pos,m,sbn,seq[m]) );
    }

    float p = exp(-(newenergy - *energy));
    if (p > rng) {
        seq[pos] = res;
        *energy = newenergy;
    }
}

void *mcmc(void *t){
    int i, j, n, m;
    uint id = *(uint*)t;
    threefry2x32_key_t k = {{0, id}};
    threefry2x32_ctr_t c = {{0xdeadbeef, getRandSeed()}};

    // correction needed if 2^32 is not a multiple of L*q
    // A bit tricky to avoid overflow: need two moduli
    uint32_t Lq_rng_lim = 0xffffffffu - (((0xffffffffu % (L*q)) + 1u) % (L*q));

    char *seq = malloc(sizeof(char)*L);
    for(i = 0; i < L; i++){
        seq[i] = sseq[i];
    }

    float energy;
    for(n = 0; n < loops; n++){
        energy = getEnergy(seq);//reset floating point error

        for (i = 0; i < steps; ) {
            c.v[0]++;
            threefry2x32_ctr_t rng = threefry2x32(c, k);

            if (rng.v[0] <= Lq_rng_lim) {
                uint32_t Lq = rng.v[0] % (L*q);
                step(&energy, seq, Lq/q, Lq%q, uniformMap32(rng.v[1]));
                i++;
            }
        }

        for(i = 0; i < L; i++){
            printf("%c", alphabet[seq[i]]);
        }
        printf("\n");
    }

    fprintf(stderr, "%20d", energy);

    free(seq);
#if NUM_THREADS > 1
    pthread_exit((void*) t);
#endif
}

char *readFile(char *filename, int *fileLen){
    FILE * f;
    long flen;
    char * buffer;

    f = fopen(filename, "rb");
    if(f == NULL){
        return NULL;
    }

    // obtain file size:
    fseek(f, 0, SEEK_END);
    flen = ftell(f);
    rewind(f);

    buffer = (char*)malloc(sizeof(char)*(flen+1));
    size_t count = fread(buffer, 1, flen, f);
    *fileLen = flen;

    fclose(f);

    return buffer;
}

int main(int argc, char *argv[]){
    int i,j,n,a,b;

    if (argc != 6) {
        fprintf(stderr, "Usage: ./a.out couplings steps loops alphabet "
                        "startseq\n");
        exit(1);
    }

    // Read in the file of couplings, which is expected to be a binary list of
    // float values. Create this from a npy file with:
    // >>> j = load('J.npy')
    // >>> j.tofile('J.dat')
    int filelen;
    float *incpl = (float*)readFile(argv[1], &filelen);
    if (filelen % 4 != 0) {
        fprintf(stderr, "Error: coupling file must contain float32. "
                        "Got %d", filelen);
        exit(1);
    }
    filelen = filelen/4;  // size in # floats

    steps = atoi(argv[2]);
    loops = atoi(argv[3]);

    alphabet = argv[4];
    q = strlen(alphabet);
    if (filelen % (q*q) != 0){
        fprintf(stderr, "Error: size of alphabet (%d) does not i"
                        "match coupling file", strlen(alphabet));
        exit(1);
    }

    char *startseq = argv[5];

    unsigned int nPairs = filelen/(q*q);
    L = (int)((1+sqrt(1+8*nPairs))/2);

    if ((L*(L-1))/2 != nPairs) {
        fprintf(stderr, "Error: coupling file did not have length L*(L-1)/2. "
                        "Got %d", nPairs);
        exit(1);
    }

    // expand couplings to full matrix (better memory access)
    couplings = malloc(sizeof(float)*L*L*q*q);
    n = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            for(a = 0; a < q; a++){
                for(b = 0; b < q; b++){
                    coupling(i,j,a,b) = incpl[q*q*n + q*a + b];
                    coupling(j,i,b,a) = incpl[q*q*n + q*a + b];
                }
            }
            n++;
        }
    }

    sseq = malloc(L);
    for(i = 0; i < L; i++){
        sseq[i] = (char)(strchr(alphabet, startseq[i]) - alphabet);
    }

    int rc;
    uint t = 0;
    void *status;


#if NUM_THREADS > 1

    pthread_t thread[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<NUM_THREADS; t++) {
        rc = pthread_create(&thread[t], &attr, mcmc, (void *)&t);
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(1);
        }
    }

    pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
        rc = pthread_join(thread[t], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(1);
        }
    }

    free(sseq);
    pthread_exit(NULL);

#else

    mcmc(&t);

#endif
}
