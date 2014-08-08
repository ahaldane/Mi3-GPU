#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "Random123/threefry.h"
#include "math.h"
#include "common/epsilons.h"
#include "common/rng.h"

unsigned int nBases, seqLen;
double *couplings;
#define field(i,a) fields[a + i*nBases]
#define coupling(i,j,a,b) couplings[b + nBases*a + nBases*nBases*(i*seqLen-i*(i+1)/2 + j-i-1)]
//assumes i < j !!!!!

double getEnergy(int *seq){
    int i, j;
    double energy = 0;
    for(i = 0; i < seqLen-1; i++){
        for(j = i+1; j < seqLen; j++){
            energy += coupling(i,j,seq[i],seq[j]);
        }
    }
    return energy;
}

//faster version that only calculates change in E, so is subject to floating point error
double newEnergy(int *seq, double energy, int pos, int residue){
    int m;
    double newenergy = energy;
    for(m = 0; m < seqLen; m++){
        if(m < pos){
            newenergy += ( coupling(m,pos,seq[m],residue) - 
                           coupling(m,pos,seq[m],seq[pos]) );
        }
        if(m > pos){
            newenergy += (coupling(pos,m,residue,seq[m]) - 
                          coupling(pos,m,seq[pos],seq[m]) );
        }
    }
    return newenergy;
}

int main(int argc, char *argv[]){
    int i,l,n;

    if(argc != 5){
        fprintf(stderr, "Usage: ./a.out couplings outputInterval "
                        "alphabet startseq\n");
        exit(1);
    }
    
    //init rng
    threefry4x32_key_t k = {{0, 1, 2, getRandSeed()}};
    threefry4x32_ctr_t c = {{0, 1, 0xdeadbeef, 0xbeeff00d}};

    unsigned int nPairs, nComb; //dummy
    couplings = readEM(&nComb, &nPairs, argv[1]);
    seqLen = (int)((1+sqrt(1+8*nPairs))/2);
    nBases = (int)(sqrt(nComb));

    int steps = atoi(argv[2]);

    char *alphabet = argv[3];
    if(strlen(alphabet) != nBases){
        fprintf(stderr, "Error: size of alphabet (%d) does not i"
                        "match nBases (%d)", strlen(alphabet), nBases);
    }
    printf("#PARAM alpha: '%s'\n", alphabet);

    char *startseq = argv[4];
    int *seq = malloc(sizeof(int)*seqLen);
    printf("#INFO Starting seq: ");
    for(i = 0; i < seqLen; i++){
        seq[i] = (int)(strchr(alphabet, startseq[i]) - alphabet);
        printf("%c", alphabet[seq[i]]);
    }
    printf("\n");
    double energy = getEnergy(seq);
    double E0 = energy;

    //FILE *outf = fopen("out", "wb"); //used to output autocorrelation data
    
    while(1){
        uint naccept = 0;
        double meandE = 0;
        double ddE = 0;
        for(n = 0; n < steps; n++){
            c.v[0]++; 
            threefry4x32_ctr_t rng = threefry4x32(c, k);

            //repeat sequence update 2x, using 2 of the 4 rngs each time
            for(i = 0; i <= 1; i += 2){
                uint r = rng.v[i]%(nBases*seqLen);
                uint pos = r/nBases;
                uint residue = r%nBases;

                double newenergy = newEnergy(seq, energy, pos, residue);
                double p = exp(-(newenergy - energy));
                
                ddE = newenergy - energy;

                if(p > uniformMap32(rng.v[i+1]) && !isinf(newenergy)){
                    meandE += newenergy - energy;
                    seq[pos] = residue;
                    energy = newenergy;
                    naccept++;
                }
            
                //for use in calculating the autocorrelation time:
                //fwrite(&energy, sizeof(double), 1, outf);
            }
        }
        fprintf(stderr, "E=%g dE=(%g, %g) acc=%g%%\n", energy, ddE, meandE/steps, naccept/((float)(steps)));
        for(l = 0; l < seqLen; l++){
            printf("%c", alphabet[seq[l]]);
        }
        printf("\n");

        energy = getEnergy(seq);//reset floating point error
    }
}
