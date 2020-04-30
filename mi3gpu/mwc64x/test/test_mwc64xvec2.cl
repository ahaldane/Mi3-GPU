#include "mwc64x.cl"

__kernel void TestRng(__global uint *success)
{
	mwc64xvec2_state_t a={(uint2)(0,0),(uint2)(1,2)}, b;
	mwc64x_state_t a32={0, 1}, b32;
	uint lSuccess=1;
	
	for(uint dist=0;(dist<4096 && lSuccess);dist++){
		b=a;
		MWC64XVEC2_Skip(&b, dist);
		MWC64X_Skip(&b32, dist);
		for(uint i=0;i<dist;i++){
			MWC64XVEC2_NextUint2(&a);
			MWC64X_NextUint(&a32);
		}
		lSuccess &= ((a.x.s0==a32.x) && (a.c.s0==a32.c));
		lSuccess &= (all(a.x==b.x) && all(a.c==b.c));
		
	}
	
	*success=lSuccess;
}

// total is the number of words to output, must be divisible by 16*get_global_dim(0)
__kernel void DumpSamples(uint total, __global uint *data)
{
	ulong perStream=total/(2*get_global_size(0));
	
	mwc64xvec2_state_t rng;
	MWC64XVEC2_SeedStreams(&rng, 0, perStream);
	
	for(uint i=0;i<perStream;i++){
		uint2 x=MWC64XVEC2_NextUint2(&rng);
		data[get_global_id(0)*2*perStream+0*perStream+i]=x.s0;
		data[get_global_id(0)*2*perStream+1*perStream+i]=x.s1;
	}
}

__kernel void EstimatePi(ulong n, ulong baseOffset, __global ulong *acc)
{
	mwc64xvec2_state_t rng;
    ulong samplesPerStream=n/(2*get_global_size(0));
    MWC64XVEC2_SeedStreams(&rng, baseOffset, 2*samplesPerStream);
    uint count=0;
    for(uint i=0;i<samplesPerStream;i++){
        ulong2 x=convert_ulong2(MWC64XVEC2_NextUint2(&rng));
        ulong2 y=convert_ulong2(MWC64XVEC2_NextUint2(&rng));
        ulong2 x2=x*x;
        ulong2 y2=y*y;
		long2 hit = -(x2+y2 >= x2);
		count += hit.x+hit.y;
    }
	acc[get_global_id(0)] = count;
}
