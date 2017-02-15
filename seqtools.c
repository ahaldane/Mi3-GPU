//Copyright 2016 Allan Haldane.
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
#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// compile me with 
// python2 ./setup.py build_ext --inplace

typedef unsigned int uint;
typedef unsigned char uint8;
typedef char int8;
typedef uint32_t uint32;
typedef uint64_t uint64;

static PyObject *
nsim(PyObject *self, PyObject *args){	
    PyArrayObject *seqs;
    PyObject *nsim;
    uint32 *nsimdat;
    uint32 *hsim, *origindex;
    uint8 *seqdata, *oldseq, *newseq;
    npy_intp dim;
    int i, j, p, nextind;
    npy_intp nseq, L;
    int simCutoff;

	if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &seqs, &simCutoff)){
		return NULL;
    }

	if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
		PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
		return NULL;
	}

    if(!PyArray_IS_C_CONTIGUOUS(seqs)){
		PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
		return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if(nseq == 0){
		PyErr_SetString( PyExc_ValueError, "no seqs supplied");
		return NULL;
    }
    
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);

    dim = nseq;
    nsim = PyArray_SimpleNew(1, &dim, NPY_UINT32);
    nsimdat = PyArray_DATA((PyArrayObject*)nsim);
    for(i = 0; i < nseq; i++){
        nsimdat[i] = 0;
    }

    hsim = malloc(sizeof(uint32)*nseq);
    origindex = malloc(sizeof(uint32)*nseq);
    newseq = malloc(L);
    oldseq = malloc(L);
    for(p = 0; p < L; p++){
        oldseq[p] = 0xff;
    }

    for(j = 0; j < nseq; j++){
        hsim[j] = 0;
        origindex[j] = j;
    }
    
    nextind = 0;
    for(i = 0; i < nseq-1; i++){
        
        // record next seq and copy first seq to position of chosen seq
        for(p = 0; p < L; p++){
            newseq[p] = seqdata[nseq*p + nextind];
            seqdata[nseq*p + nextind] = seqdata[nseq*p + i];
        }
        hsim[nextind] = hsim[i];
        // swap originindex
        uint tmp_originindex = origindex[i];
        origindex[i] = origindex[nextind];
        origindex[nextind] = tmp_originindex;
        // swap nsimdat
        uint tmp_nsimdat = nsimdat[i];
        nsimdat[i] = nsimdat[nextind];
        nsimdat[nextind] = tmp_nsimdat;
        
        for(p = 0; p < L; p++){
            uint8 newc = newseq[p];
            uint8 oldc = oldseq[p];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }

            for(j = i+1; j < nseq; j++){
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
        }
        
        nextind = i+1;
        uint32 biggesthsim = 0;
        for(j = i+1; j < nseq; j++){
            if(hsim[j] > biggesthsim){
                nextind = j;
                biggesthsim = hsim[j];
            }
            
            if(L-hsim[j] < simCutoff){
                nsimdat[j]++;
                nsimdat[i]++;
            }
        }
        nsimdat[i]++;

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }
    nsimdat[nseq-1]++;

    // put each element where it should go
    for(i = 0; i < nseq; i++){ 
        int ind = origindex[i];
        uint32 val = nsimdat[i];
        while(ind != i){
            int newind = origindex[ind];
            uint32 tmpval = nsimdat[ind];
            nsimdat[ind] = val;
            origindex[ind] = ind;
            ind = newind;
            val = tmpval;
        }
        nsimdat[i] = val;
    }
    
    free(oldseq);
    free(newseq);
    free(hsim);
    free(origindex);
    Py_DECREF(seqs);
    return nsim;
}

static PyObject *
histsim(PyObject *self, PyObject *args){	
    PyArrayObject *seqs;
    PyObject *hist;
    uint64 *histdat;
    uint32 *hsim;
    uint8 *seqdata, *oldseq, *newseq;
    npy_intp dim;
    int i, j, p, nextind;
    npy_intp nseq, L;

	if(!PyArg_ParseTuple(args, "O!", &PyArray_Type, &seqs)){
		return NULL;
    }

	if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
		PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
		return NULL;
	}

    if(!PyArray_IS_C_CONTIGUOUS(seqs)){
		PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
		return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if(nseq == 0){
		PyErr_SetString( PyExc_ValueError, "no seqs supplied");
		return NULL;
    }
    
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);

    dim = L+1;
    hist = PyArray_SimpleNew(1, &dim, NPY_UINT64);
    histdat = PyArray_DATA((PyArrayObject*)hist);
    for(p = 0; p < L+1; p++){
        histdat[p] = 0;
    }

    hsim = malloc(sizeof(uint32)*nseq);
    newseq = malloc(L);
    oldseq = malloc(L);
    for(p = 0; p < L; p++){
        oldseq[p] = 0xff;
    }

    for(j = 0; j < nseq; j++){
        hsim[j] = 0;
    }
    
    nextind = 0;
    for(i = 0; i < nseq-1; i++){
        
        // record next seq and copy first seq to position of chosen seq
        for(p = 0; p < L; p++){
            newseq[p] = seqdata[nseq*p + nextind];
            seqdata[nseq*p + nextind] = seqdata[nseq*p + i];
        }
        hsim[nextind] = hsim[i];
        
        for(p = 0; p < L; p++){
            uint8 newc = newseq[p];
            uint8 oldc = oldseq[p];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }
            
            for(j = i+1; j < nseq; j++){
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
        }
        
        nextind = i+1;
        uint32 biggesthsim = 0;
        for(j = i+1; j < nseq; j++){
            if(hsim[j] > biggesthsim){
                nextind = j;
                biggesthsim = hsim[j];
            }
            histdat[hsim[j]]++;
        }
        histdat[L]++; //self term

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }
    histdat[L]++;
    
    free(oldseq);
    free(newseq);
    free(hsim);
    Py_DECREF(seqs);
    return hist;
}
/*
 * Helper function for sequence loading, which converts the sequences to
 * integer format. Takes in an array of (nseq, L+1) bytes arranged like in the
 * file (ascii + newline at the end) and overwrites the ascii with integers,
 * while also doing sanity checks.
 */
static PyObject *
translateascii(PyObject *self, PyObject *args){	
    PyArrayObject *seqs;
    uint8 *seqdata;
    npy_intp nseq, L;
    uint8 *alpha;
    int i, j, offset;
    uint8 translationtable[256];

	if(!PyArg_ParseTuple(args, "O!si", &PyArray_Type, &seqs, &alpha, &offset)){
		return NULL;
    }

	if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
		PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
		return NULL;
	}

    if(!PyArray_IS_C_CONTIGUOUS(seqs)){
		PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
		return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1)-1;
    seqdata = PyArray_DATA(seqs);

    for(i = 0; i < 256; i++){
        translationtable[i] = 0xff;
    }
    for(i = 0; i < strlen((char*)alpha); i++){
        if(alpha[i] == 0xff){
            PyErr_SetString(PyExc_ValueError, "Alphabet cannot contain 0xff");
            return NULL;
        }
        translationtable[alpha[i]] = i;
    }

    for(i = 0; i < nseq; i++){
        for(j = 0; j < L; j++){
            uint8 newc = translationtable[seqdata[i*(L+1) + j]];
            if(newc == 0xff){
                if(seqdata[i*(L+1) + j] == '\n'){
                    PyErr_Format(PyExc_ValueError, 
                       "Sequence %d has length %d (expected %d)", 
                       i+offset, j, (int)L);
                }
                else{
                    PyErr_Format(PyExc_ValueError, 
                        "Invalid Residue '%c' in sequence %d position %d", 
                        seqdata[i*(L+1) + j], i+offset, j+1);
                }
                return NULL;
            }
            seqdata[i*(L+1) + j] = newc;
        }
        if(seqdata[i*(L+1) + L] != '\n'){
            PyErr_Format(PyExc_ValueError, 
                       "Sequence %d has length %d (expected %d)", 
                       i+offset, j, (int)L);
            return NULL;
        }
    }
    
    Py_RETURN_NONE;
}

static PyMethodDef SeqtoolsMethods[] = {
	{"nsim", nsim, METH_VARARGS, 
            "compute number of similar sequences"},
	{"histsim", histsim, METH_VARARGS, 
            "histogram pariwise similarities"},
	{"translateascii", translateascii, METH_VARARGS, 
            "translate sequence buffer from scii to integers"},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC
initseqtools(void){
    Py_InitModule("seqtools", SeqtoolsMethods);
    import_array();
}

/* Below is an older version of nsim which uses the hamming distance
 * triangle inequalities to speed up the computation. It turns out to
 * be slower than the implementation above for typical datasets, but
 * I leave it here since it was a fun idea and it might be useful
 * somehow in the future.
*/

/*
#define min(x,y)  (((x) < (y)) ? (x) : (y))

int 
hamming(uint8 *seq1, uint8 *seq2, int L){
    int i, h = 0;
    for(i = 0; i < L; i++){
        if(seq1[i] != seq2[i]){
            h++;
        }
    }
    return h;
}

void 
swap(int *arr, int i1, int i2){
    int temp = arr[i1];
    arr[i1] = arr[i2];
    arr[i2] = temp;
}

static PyObject *
nsim(PyObject *self, PyObject *args){	
    PyArrayObject *nsim, *seqs;
    uint8 *seqd;
    int i, j, si, sj, lowest, iip1dist, nextseq=0;
    npy_intp nseq, L;
    int simCutoff;
    int *upperbounds, *lowerbounds, *nsimd;
    int *seqinds;

	if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &seqs, &simCutoff)){
		return NULL;
    }

	if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
		PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
		return NULL;
	}
    
    seqs = PyArray_GETCONTIGUOUS(seqs);
    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);
    seqd = PyArray_DATA(seqs);

    upperbounds = malloc(sizeof(int)*nseq);
    lowerbounds = malloc(sizeof(int)*nseq);
    seqinds = malloc(sizeof(int)*nseq);
    nsim = (PyArrayObject*)PyArray_ZEROS(1, &nseq, NPY_INT, 0);
    nsimd = PyArray_DATA(nsim);

    // use bounds from triangle inequalities to skip unneeded comparisons

    for(i = 0; i < nseq; i++){
        upperbounds[i] = L;
        lowerbounds[i] = 0;
        seqinds[i] = i;
    }
    
    for(i = 0; i < nseq; i++){
        si = seqinds[i];
        nsimd[si]++;
        lowest = INT_MAX;
        for(j = i+1; j < nseq; j++){
            sj = seqinds[j];
            if(lowerbounds[j] < simCutoff){
                if(upperbounds[j] >= simCutoff){
                    int h = hamming(&seqd[si*L], &seqd[sj*L], L);
                    lowerbounds[j] = h;
                    upperbounds[j] = h;
                }

                if(upperbounds[j] < simCutoff){
                    nsimd[si]++;
                    nsimd[sj]++;
                }
            }
            
            // for next round, choose a similar reference sequence
            if(lowerbounds[j] < lowest){
                nextseq = j;
                lowest = lowerbounds[j];
            }
        }

        iip1dist = hamming(&seqd[si*L], &seqd[seqinds[nextseq]*L], L);

        // only keep track of nseq - i sequences. Do this by moving
        // nextseq to the start of tracked sequences (position i)
        swap(    seqinds, nextseq, i+1);
        swap(lowerbounds, nextseq, i+1);
        swap(upperbounds, nextseq, i+1);
        
        for(j = i+2; j < nseq; j++){
            upperbounds[j] += iip1dist;
            lowerbounds[j] -= iip1dist;
        }
    }

    free(lowerbounds);
    free(upperbounds);
    free(seqinds);
    Py_DECREF(seqs);
    
    return (PyObject*)nsim;
}

*/

/* Note:
 The lower bound update step above, "lowerbounds[j] -= iip1dist", can be
 tightened a little, but it turns out to only give a small reduction
 in # of hamming distances computed, so isn't worth it. Tighter lower bound:

            if(iip1dist < lowerbounds[j]){
                lowerbounds[j] = lowerbounds[j] - iip1dist;
            }
            else if(iip1dist > upperbounds[j]){
                lowerbounds[j] = iip1dist - upperbounds[j];
            }
            else{
                lowerbounds[j] = 0;
            }
*/
