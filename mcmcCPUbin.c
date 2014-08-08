#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Random123/threefry.h"
#include "math.h"
#include "common/epsilons.h"
#include "common/rng.h"

typedef unsigned int uint;
typedef unsigned char uchar;

uint nB, L;
double *couplings;
#define IND(i,j,a,b) (b + nB*a + nB*nB*(i*L-i*(i+1)/2 + j-i-1))
//assumes i < j !!!!!

double getEnergy(int *seq){
    int i, j;
    double energy = 0;
    for(i = 0; i < L-1; i++){
        for(j = i+1; j < L; j++){
            energy += couplings[IND(i,j,seq[i],seq[j])];
        }
    }
    return energy;
}

//faster version that only calculates change in E, so is subject to floating point error
double newEnergy(int *seq, double energy, int pos, int residue){
    int m;
    double newenergy = energy;
    for(m = 0; m < L; m++){
        if(m < pos){
            newenergy += ( couplings[IND(m,pos,seq[m],residue)] - 
                           couplings[IND(m,pos,seq[m],seq[pos])] );
        }
        if(m > pos){
            newenergy += (couplings[IND(pos,m,residue,seq[m])] - 
                          couplings[IND(pos,m,seq[pos],seq[m])] );
        }
    }
    return newenergy;
}

int main(int argc, char *argv[]){
    int i,j,l,n;

    if(argc != 7){
        fprintf(stderr, "Usage: ./a.out L nB loops steps seed groupseed\n"
            "Reads 64bit couplings on stdin, writes 64bit bicounts on stdout\n");
            //same endiannes as system
        exit(1);
    }
    
    L = atoi(argv[1]);
    nB = atoi(argv[2]);
    uint nloop = atoi(argv[3]);
    uint steps = atoi(argv[4]);
    uint seed = atoi(argv[5]);
    uint groupseed = atoi(argv[6]);

    freopen(NULL, "rb", stdin);
    freopen(NULL, "wb", stdout);
       
    uint nPairs = L*(L-1)/2;
    uint nComb = nB*nB;
    uint nCouplings = nPairs*nComb;
    
    //init rng
    threefry4x32_key_t k = {{0, 1, groupseed, seed}};
    threefry4x32_ctr_t c = {{0, 1, 0xdeadbeef, 0xbeeff00d}};
    int *seq = malloc(sizeof(int)*L);
    uchar *cseq = malloc(sizeof(uchar)*L);
    fread_ordie(cseq, sizeof(uchar), L, stdin);
    for(i = 0; i < L; i++){
        seq[i] = cseq[i];
    }

    couplings = malloc(sizeof(double)*nCouplings);
    fread_ordie(couplings, sizeof(double), nCouplings, stdin);
    uint64_t *bicounts = malloc(sizeof(uint64_t)*nPairs*nComb);
    for(i = 0; i < nCouplings; i++){
        bicounts[i] = 0;
    }
    double energy = getEnergy(seq);


    for(l = 0; l < nloop; l++){
        for(n = 0; n < steps; n++){
            c.v[0]++; 
            threefry4x32_ctr_t rng = threefry4x32(c, k);

            //repeat sequence update 2x, using 2 of the 4 rngs each time
            for(i = 0; i <= 1; i += 2){
                uint r = rng.v[i]%(nB*L);
                uint pos = r/nB;
                uint residue = r%nB;

                double newenergy = newEnergy(seq, energy, pos, residue);
                double p = exp(-(newenergy - energy));
                if(p > uniformMap32(rng.v[i+1]) && !isinf(newenergy)){
                    seq[pos] = residue;
                    energy = newenergy;
                }
            }
        }

        //count
        for(i = 0; i < L-1; i++){
            for(j = i+1; j < L; j++){
                bicounts[IND(i,j,seq[i],seq[j])]++;
            }
        }

        energy = getEnergy(seq);//reset floating point error
    }

    for(i = 0; i < L; i++){
        cseq[i] = seq[i];
    }
    fwrite(cseq, sizeof(uchar), L, stdout);
    fwrite(bicounts, sizeof(uint64_t), nCouplings, stdout);
}
