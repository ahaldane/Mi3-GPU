#include "mwc64x.cl"

__kernel void TestRng(__global uint *success)
{
	mwc64xvec8_state_t a={(uint8)(0,0,0,0,0,0,0,0),(uint8)(1,2,3,4,5,6,7,8)}, b;
	mwc64x_state_t a32={0, 1}, b32;
	uint lSuccess=1;
	
	for(uint dist=0;(dist<4096 && lSuccess);dist++){
		b=a;
		MWC64XVEC8_Skip(&b, dist);
		MWC64X_Skip(&b32, dist);
		for(uint i=0;i<dist;i++){
			MWC64XVEC8_NextUint8(&a);
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
	ulong perStream=total/(8*get_global_size(0));
	
	mwc64xvec8_state_t rng;
	MWC64XVEC8_SeedStreams(&rng, 0, perStream);
	
	for(uint i=0;i<perStream;i++){
		uint8 x=MWC64XVEC8_NextUint8(&rng);
		data[get_global_id(0)*8*perStream+0*perStream+i]=x.s0;
		data[get_global_id(0)*8*perStream+1*perStream+i]=x.s1;
		data[get_global_id(0)*8*perStream+2*perStream+i]=x.s2;
		data[get_global_id(0)*8*perStream+3*perStream+i]=x.s3;
		data[get_global_id(0)*8*perStream+4*perStream+i]=x.s4;
		data[get_global_id(0)*8*perStream+5*perStream+i]=x.s5;
		data[get_global_id(0)*8*perStream+6*perStream+i]=x.s6;
		data[get_global_id(0)*8*perStream+7*perStream+i]=x.s7;
	}
}

__kernel void EstimatePi(ulong n, ulong baseOffset, __global ulong *acc)
{
	mwc64xvec8_state_t rng;
    ulong samplesPerStream=n/(8*get_global_size(0));
    MWC64XVEC8_SeedStreams(&rng, baseOffset, 2*samplesPerStream);
    uint count=0;
    for(uint i=0;i<samplesPerStream;i++){
        ulong8 x=convert_ulong8(MWC64XVEC8_NextUint8(&rng));
        ulong8 y=convert_ulong8(MWC64XVEC8_NextUint8(&rng));
        ulong8 x2=x*x;
        ulong8 y2=y*y;
		long8 hit = -(x2+y2 >= x2);
		count += hit.s0+hit.s1+hit.s2+hit.s3 + hit.s4+hit.s5+hit.s6+hit.s7;
    }
	acc[get_global_id(0)] = count;
}
