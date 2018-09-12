//Copyright 2018 Allan Haldane.
//
//This file is part of IvoGPU.
//
//IvoGPU is free software: you can redistribute it and/or modify
//it under the terms of the GNU General Public License as published by
//the Free Software Foundation, version 3 of the License.
//
//IvoGPU is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with IvoGPU.  If not, see <http://www.gnu.org/licenses/>.
//
//Contact: allan.haldane _AT_ gmail.com
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include "Random123/threefry.h"
#include "math.h"
#include "common/epsilons.h"
#include "common/rng.h"
#define NUM_THREADS 8

unsigned int nB, L;
float *couplings;
#define coupling(i,j,a,b) couplings[(L*i + j)*nB*nB + a*nB + b]
char *alphabet;
char *sseq;
uint loops, steps;

double getEnergy(char *seq){
    int i, j;
    double energy = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            energy += coupling(i,j,seq[i],seq[j]);
        }
    }
    return energy;
}

void *mcmc(void *t){
    int i, j, n, m;
    uint id = (int)t;
    threefry4x32_key_t k = {{0, 1, 2, id}};
    threefry4x32_ctr_t c = {{0, 1, 0xdeadbeef, 0xbeeff00d}};

    char *seq = malloc(sizeof(char)*L);
    for(i = 0; i < L; i++){
        seq[i] = sseq[i];
    }
    
    float energy;
    for(n = 0; n < loops; n++){
        energy = getEnergy(seq);//reset floating point error
        uint pos = 0;
        for(i = 0; i < steps; i++){
            c.v[0]++; 
            threefry4x32_ctr_t rng = threefry4x32(c, k);
            uint residue = rng.v[0]%nB;

            double newenergy = energy;
            uint sbn = seq[pos];
            for(m = 0; m < L; m++){
                newenergy += (coupling(pos,m, residue,seq[m]) - 
                              coupling(pos,m,seq[pos],seq[m]) );
            }
                
            float p = exp(-(newenergy - energy));
            if(p > uniformMap32(rng.v[1]) && !isinf(newenergy)){
                seq[pos] = residue;
                energy = newenergy;
            }

            pos = (pos+1)%L;
        }
    }

    for(i = 0; i < L; i++){
        printf("%c", alphabet[seq[i]]);
    }
    printf("  %g \n", energy);

    free(seq);
    pthread_exit((void*) t);
}

int main(int argc, char *argv[]){
    int i,j,n,a,b;

    if(argc != 6){
        fprintf(stderr, "Usage: ./a.out couplings steps loops alphabet startseq\n");
        exit(1);
    }

    unsigned int nPairs, nComb; 
    double *incpl = readEM(&nComb, &nPairs, argv[1]);
    L = (int)((1+sqrt(1+8*nPairs))/2);
    nB = (int)(sqrt(nComb));
    
    // expand couplings to full matrix (better memory access)
    couplings = malloc(sizeof(float)*L*L*nB*nB);
    n = 0;
    for(i = 0; i < L-1; i++){
    for(j = i+1; j < L; j++){
        for(a = 0; a < nB; a++){
        for(b = 0; b < nB; b++){
            coupling(i,j,a,b) = incpl[nB*nB*n + nB*a + b];
            coupling(j,i,b,a) = incpl[nB*nB*n + nB*a + b];
        }
        }
        n++;
    }
    }

    steps = atoi(argv[2]);
    loops = atoi(argv[3]);

    alphabet = argv[4];
    if(strlen(alphabet) != nB){
        fprintf(stderr, "Error: size of alphabet (%d) does not i"
                        "match nB (%d)", strlen(alphabet), nB);
    }
    printf("#PARAM alpha: '%s'\n", alphabet);

    char *startseq = argv[5];
    sseq = malloc(sizeof(char)*L);
    printf("#INFO Starting seq: ");
    for(i = 0; i < L; i++){
        sseq[i] = (char)(strchr(alphabet, startseq[i]) - alphabet);
        printf("%c", alphabet[sseq[i]]);
    }
    printf("\n");

    int rc;
    long t;
    void *status;

    pthread_t thread[NUM_THREADS];
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    for(t=0; t<NUM_THREADS; t++) {
        printf("Main: creating thread %ld\n", t);
        rc = pthread_create(&thread[t], &attr, mcmc, (void *)t); 
        if (rc) {
            printf("ERROR; return code from pthread_create() is %d\n", rc);
            exit(-1);
        }
    }

    pthread_attr_destroy(&attr);
    for(t=0; t<NUM_THREADS; t++) {
        rc = pthread_join(thread[t], &status);
        if (rc) {
            printf("ERROR; return code from pthread_join() is %d\n", rc);
            exit(-1);
        }
        printf("Main: completed join with thread %ld having a status of %ld\n",t,(long)status);
    }

    printf("Main: program completed. Exiting.\n");
    free(sseq);
    pthread_exit(NULL);
}
