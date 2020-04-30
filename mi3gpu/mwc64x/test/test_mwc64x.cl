#include "mwc64x.cl"

__kernel void TestRng(__global uint *success)
{
	mwc64x_state_t a={0,1}, b;
	uint lSuccess=1;
	
	for(uint dist=0;(dist<4096 && lSuccess);dist++){
		b=a;
		MWC64X_Skip(&b, dist);
		for(uint i=0;i<dist;i++){
			MWC64X_NextUint(&a);
		}
		lSuccess &= ((a.x==b.x) && (a.c==b.c));
	}
	
	*success=lSuccess;
}

// total is the number of words to output, must be divisible by 16*get_global_dim(0)
__kernel void DumpSamples(uint total, __global uint *data)
{
	ulong perStream=total/get_global_size(0);
	
	mwc64x_state_t rng;
	MWC64X_SeedStreams(&rng, 0, perStream);
	
	__global uint *dest=data+get_global_id(0)*perStream;
	for(uint i=0;i<perStream;i++){
		dest[i]=MWC64X_NextUint(&rng);
	}
}

__kernel void EstimatePi(ulong n, ulong baseOffset, __global ulong *acc)
{
	mwc64x_state_t rng;
    ulong samplesPerStream=n/get_global_size(0);
    MWC64X_SeedStreams(&rng, baseOffset, 2*samplesPerStream);
    uint count=0;
    for(uint i=0;i<samplesPerStream;i++){
        ulong x=MWC64X_NextUint(&rng);
        ulong y=MWC64X_NextUint(&rng);
        ulong x2=x*x;
        ulong y2=y*y;
        if(x2+y2 >= x2)
            count++;
    }
	acc[get_global_id(0)] = count;
}
