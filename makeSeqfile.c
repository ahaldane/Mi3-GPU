#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define WGSIZE 256
#define NGROUPS 32

typedef unsigned char uchar;
typedef unsigned int uint;

int main(int argc, char *argv[]){
    if(argc != 3 && argc != 4){
        printf("Usage: ./a.out alphabet startseq [outfile]");
        exit(1);
    }
     int i,n;
    char *alphabet = argv[1];
    char *seqstr = argv[2];
    char *outfn = "sequences";
    if(argc == 4){
        outfn = argv[3]; 
    }
    
    uint nB = strlen(alphabet);
    uint L = strlen(seqstr);
    printf("L = %d  nB = %d\n", L, nB);
    printf("Generating for WGSIZE %d and NGROUPS %d\n", WGSIZE, NGROUPS);
    
    //pad sequenes to 32 bit boundaries
    uint swords = (L-1)/4+1;
    uint sbytes = 4*swords;

    uint *startseqmem = malloc(sizeof(uint)*swords);
    for(i = 0; i < swords; i++){
        startseqmem[i] = 0;
    }
    uchar *startseq = (uchar*)startseqmem;
    for(i = 0; i < L; i++){
        char *cpos = strchr(alphabet, argv[2][i]);
        if(cpos == NULL){
            printf("Invalid letter: %c\n", argv[2][i]);
            exit(1);
        }
        startseq[i] = (uchar)(cpos - alphabet); 
    }

    uint *seqmem = malloc(sizeof(uint)*swords*WGSIZE*NGROUPS);
    for(i = 0; i < swords; i++){
        for(n = 0; n < WGSIZE*NGROUPS; n++){
            seqmem[i*NGROUPS*WGSIZE + n] = startseqmem[i];
        }
    }
    
    uint wgsize = WGSIZE;
    uint ngroups = NGROUPS;

    FILE *f = fopen(outfn, "wb");
    fwrite(&wgsize, sizeof(uint), 1, f);
    fwrite(&ngroups, sizeof(uint), 1, f);
    fwrite(&L, sizeof(uint), 1, f);
    fwrite(&nB, sizeof(uint), 1, f);
    fwrite(seqmem, sizeof(uint), WGSIZE*NGROUPS*swords, f);
    fclose(f);
}
