//Copyright 2020 Allan Haldane.
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
#include "Python.h"
#include <stdlib.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

// compile me with
// python ./setup.py build_ext --inplace

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

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    // transpose for optimal memory order (see below)
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

            // bottleneck of function, sequence transpose speeds this up
            for(j = i+1; j < nseq; j++){
                // using vectorizable arithmetic operations also speeds it up
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
            // if this needs to be _really_ fast, could use pthreads
            // (one thread per p)
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
nsim_weighted(PyObject *self, PyObject *args){
    PyArrayObject *seqs, *weights;
    PyObject *nsim;
    npy_float64 *nsimdat;
    uint32 *hsim, *origindex;
    uint8 *seqdata, *oldseq, *newseq;
    npy_intp dim;
    int i, j, p, nextind;
    npy_intp nseq, L;
    int simCutoff;
    npy_float64 *weightdata;

    if(!PyArg_ParseTuple(args, "O!iO!", &PyArray_Type, &seqs, &simCutoff,
                                       &PyArray_Type, &weights)){
        return NULL;
    }

    if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
        PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
        return NULL;
    }

    if( (PyArray_NDIM(weights) != 1) || PyArray_TYPE(weights) != NPY_FLOAT64 ){
        PyErr_SetString(PyExc_ValueError, "weights must be 1d float64 array");
        return NULL;
    }

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString(PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    if(!PyArray_ISCARRAY(weights)){
        PyErr_SetString(PyExc_ValueError, "weights must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);
    if(nseq != PyArray_DIM(weights,0)){
        PyErr_SetString(PyExc_ValueError,
                        "number of weights must equal number of sequences");
        return NULL;
    }

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);
    weightdata = PyArray_DATA(weights);

    dim = nseq;
    nsim = PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    nsimdat = PyArray_DATA((PyArrayObject*)nsim);
    for(i = 0; i < nseq; i++){
        nsimdat[i] = 0;
        if(weightdata[i] == 0){ printf("X1");}
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
        npy_float64 tmp_nsimdat = nsimdat[i];
        nsimdat[i] = nsimdat[nextind];
        nsimdat[nextind] = tmp_nsimdat;
        // swap weights
        npy_float64 tmp_weight = weightdata[i];
        weightdata[i] = weightdata[nextind];
        weightdata[nextind] = tmp_weight;

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
                nsimdat[j] += weightdata[i];
                nsimdat[i] += weightdata[j];
            }
        }
        nsimdat[i] += weightdata[i];

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }
    nsimdat[nseq-1] += weightdata[nseq-1];

    // put each element where it should go
    for(i = 0; i < nseq; i++){
        int ind = origindex[i];
        npy_float64 nsimdat_val = nsimdat[i];
        npy_float64 weight_val = weightdata[i];
        while(ind != i){
            int newind = origindex[ind];
            npy_float64 tmp_nsimdat = nsimdat[ind];
            npy_float64 tmp_weight = weightdata[ind];
            nsimdat[ind] = nsimdat_val;
            weightdata[ind] = weight_val;
            origindex[ind] = ind;

            ind = newind;
            nsimdat_val = tmp_nsimdat;
            weight_val = tmp_weight;
        }
        nsimdat[i] = nsimdat_val;
        weightdata[i] = weight_val;
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

    if(!PyArray_ISCARRAY(seqs)){
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

static PyObject *
histsim_weighted(PyObject *self, PyObject *args){
    PyArrayObject *seqs, *weights;
    PyObject *hist;
    npy_float64 *histdat;
    uint32 *hsim;
    npy_float64 *weightdata;
    uint8 *seqdata, *oldseq, *newseq;
    npy_intp dim;
    int i, j, p, nextind;
    npy_intp nseq, L;

    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &seqs,
                                       &PyArray_Type, &weights)){
        return NULL;
    }

    if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
        PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
        return NULL;
    }

    if( (PyArray_NDIM(weights) != 1) || PyArray_TYPE(weights) != NPY_FLOAT64 ){
        PyErr_SetString( PyExc_ValueError, "weights must be 1d float64 array");
        return NULL;
    }

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    if(!PyArray_ISCARRAY(weights)){
        PyErr_SetString( PyExc_ValueError, "weights must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);
    if(nseq != PyArray_DIM(weights,0)){
        PyErr_SetString( PyExc_ValueError,
            "number of weights must equal number of sequences");
        return NULL;
    }

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);
    weights = (PyArrayObject*)PyArray_Copy(weights); //clobbered below
    weightdata = PyArray_DATA(weights);

    dim = L+1;
    hist = PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
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
        npy_float64 w = weightdata[nextind];
        weightdata[nextind] = weightdata[i];
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
            histdat[hsim[j]] += weightdata[j]*w;
        }
        histdat[L] += w*w; //self term

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }
    histdat[L] += weightdata[nextind]*weightdata[nextind];

    free(oldseq);
    free(newseq);
    free(hsim);
    Py_DECREF(seqs);
    Py_DECREF(weights);
    return hist;
}

static PyObject *
minsim(PyObject *self, PyObject *args){
    PyArrayObject *seqs;
    PyObject *minsim, *minsimind;
    npy_uint32 *minsimdat;
    npy_intp *minsiminddat;
    uint32 *hsim, *origindex;
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

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    // transpose for optimal memory order (see below)
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);

    dim = nseq;
    minsim = PyArray_SimpleNew(1, &dim, NPY_UINT32);
    minsimdat = PyArray_DATA((PyArrayObject*)minsim);
    minsimind = PyArray_SimpleNew(1, &dim, NPY_INTP);
    minsiminddat = PyArray_DATA((PyArrayObject*)minsimind);
    for(i = 0; i < nseq; i++){
        minsimdat[i] = L+2;
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
        uint32 tmp_originindex = origindex[i];
        origindex[i] = origindex[nextind];
        origindex[nextind] = tmp_originindex;
        // swap minsimdat
        npy_uint32 tmp_minsimdat = minsimdat[i];
        minsimdat[i] = minsimdat[nextind];
        minsimdat[nextind] = tmp_minsimdat;
        // swap minsiminddat
        npy_intp tmp_minsiminddat = minsiminddat[i];
        minsiminddat[i] = minsiminddat[nextind];
        minsiminddat[nextind] = tmp_minsiminddat;

        for(p = 0; p < L; p++){
            uint8 newc = newseq[p];
            uint8 oldc = oldseq[p];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }

            // bottleneck of function, sequence transpose speeds this up
            for(j = i+1; j < nseq; j++){
                // using vectorizable arithmetic operations also speeds it up
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
            // if this needs to be _really_ fast, could use pthreads
            // (one thread per p)
        }

        nextind = i+1;
        uint32 biggesthsim = 0;
        for(j = i+1; j < nseq; j++){
            if(hsim[j] > biggesthsim){
                nextind = j;
                biggesthsim = hsim[j];
            }

            if (hsim[j] < minsimdat[j]) {
                minsimdat[j] = hsim[j];
                minsiminddat[j] = origindex[i];
            }

            if (hsim[j] < minsimdat[i]) {
                minsimdat[i] = hsim[j];
                minsiminddat[i] = origindex[j];
            }
        }

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }

    // put each element where it should go
    for(i = 0; i < nseq; i++){
        int ind = origindex[i];
        npy_uint32 minval = minsimdat[i];
        npy_intp minind = minsiminddat[i];
        while(ind != i){
            int newind = origindex[ind];

            npy_uint32 tmpminval = minsimdat[ind];
            npy_intp tmpminind = minsiminddat[ind];
            minsimdat[ind] = minval;
            minsiminddat[ind] = minind;

            origindex[ind] = ind;
            ind = newind;

            minval = tmpminval;
            minind = tmpminind;
        }
        minsimdat[i] = minval;
        minsiminddat[i] = minind;
    }

    free(oldseq);
    free(newseq);
    free(hsim);
    free(origindex);
    Py_DECREF(seqs);
    return PyTuple_Pack(2, minsim, minsimind);
}

static PyObject *
sumsim(PyObject *self, PyObject *args){
    PyArrayObject *seqs;
    PyObject *sumsim;
    npy_uint64 *sumsimdat;
    uint32 *hsim, *origindex;
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

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    // transpose for optimal memory order (see below)
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);

    dim = nseq;
    sumsim = PyArray_SimpleNew(1, &dim, NPY_UINT64);
    sumsimdat = PyArray_DATA((PyArrayObject*)sumsim);
    for(i = 0; i < nseq; i++){
        sumsimdat[i] = 0;
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
        uint32 tmp_originindex = origindex[i];
        origindex[i] = origindex[nextind];
        origindex[nextind] = tmp_originindex;
        // swap sumsimdat
        npy_uint64 tmp_sumsimdat = sumsimdat[i];
        sumsimdat[i] = sumsimdat[nextind];
        sumsimdat[nextind] = tmp_sumsimdat;

        for(p = 0; p < L; p++){
            uint8 newc = newseq[p];
            uint8 oldc = oldseq[p];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }

            // bottleneck of function, sequence transpose speeds this up
            for(j = i+1; j < nseq; j++){
                // using vectorizable arithmetic operations also speeds it up
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
            // if this needs to be _really_ fast, could use pthreads
            // (one thread per p)
        }

        nextind = i+1;
        uint32 biggesthsim = 0;
        for(j = i+1; j < nseq; j++){
            if(hsim[j] > biggesthsim){
                nextind = j;
                biggesthsim = hsim[j];
            }

            sumsimdat[j] += hsim[j];
            sumsimdat[i] += hsim[j];
        }
        // note: self term is not included!

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }

    // put each element where it should go
    for(i = 0; i < nseq; i++){
        int ind = origindex[i];
        npy_uint64 val = sumsimdat[i];
        while(ind != i){
            int newind = origindex[ind];

            npy_uint64 tmpval = sumsimdat[ind];
            sumsimdat[ind] = val;

            origindex[ind] = ind;
            ind = newind;

            val = tmpval;
        }
        sumsimdat[i] = val;
    }

    free(oldseq);
    free(newseq);
    free(hsim);
    free(origindex);
    Py_DECREF(seqs);
    return sumsim;
}

static PyObject *
sumsim_weighted(PyObject *self, PyObject *args){
    PyArrayObject *seqs, *weights;
    PyObject *sumsim;
    npy_float64 *sumsimdat;
    uint32 *hsim, *origindex;
    npy_float64 *weightdata;
    uint8 *seqdata, *oldseq, *newseq;
    npy_intp dim;
    int i, j, p, nextind;
    npy_intp nseq, L;

    if(!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &seqs,
                                       &PyArray_Type, &weights)){
        return NULL;
    }

    if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
        PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
        return NULL;
    }

    if( (PyArray_NDIM(weights) != 1) || PyArray_TYPE(weights) != NPY_FLOAT64 ){
        PyErr_SetString( PyExc_ValueError, "weights must be 1d float64 array");
        return NULL;
    }

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    if(!PyArray_ISCARRAY(weights)){
        PyErr_SetString( PyExc_ValueError, "weights must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);
    if(nseq != PyArray_DIM(weights,0)){
        PyErr_SetString( PyExc_ValueError,
            "number of weights must equal number of sequences");
        return NULL;
    }

    if(nseq == 0){
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    // transpose for optimal memory order (see below)
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    seqdata = PyArray_DATA(seqs);
    weights = (PyArrayObject*)PyArray_Copy(weights); //clobbered below
    weightdata = PyArray_DATA(weights);

    dim = nseq;
    sumsim = PyArray_SimpleNew(1, &dim, NPY_FLOAT64);
    sumsimdat = PyArray_DATA((PyArrayObject*)sumsim);
    for(i = 0; i < nseq; i++){
        sumsimdat[i] = 0;
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
        uint32 tmp_originindex = origindex[i];
        origindex[i] = origindex[nextind];
        origindex[nextind] = tmp_originindex;
        // swap sumsimdat
        npy_float64 tmp_sumsimdat = sumsimdat[i];
        sumsimdat[i] = sumsimdat[nextind];
        sumsimdat[nextind] = tmp_sumsimdat;
        // update weightdata
        npy_float64 w = weightdata[nextind];
        weightdata[nextind] = weightdata[i];

        for(p = 0; p < L; p++){
            uint8 newc = newseq[p];
            uint8 oldc = oldseq[p];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }

            // bottleneck of function, sequence transpose speeds this up
            for(j = i+1; j < nseq; j++){
                // using vectorizable arithmetic operations also speeds it up
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
            // if this needs to be _really_ fast, could use pthreads
            // (one thread per p)
        }

        nextind = i+1;
        uint32 biggesthsim = 0;
        for(j = i+1; j < nseq; j++){
            if(hsim[j] > biggesthsim){
                nextind = j;
                biggesthsim = hsim[j];
            }

            sumsimdat[j] += weightdata[j]*w*hsim[j];
            sumsimdat[i] += weightdata[j]*w*hsim[j];
        }
        // note: self term is not included!

        //swap sequence buffers
        uint8 *tmp = oldseq;
        oldseq = newseq;
        newseq = tmp;
    }

    // put each element where it should go
    for(i = 0; i < nseq; i++){
        int ind = origindex[i];
        npy_float64 val = sumsimdat[i];
        while(ind != i){
            int newind = origindex[ind];

            npy_float64 tmpval = sumsimdat[ind];
            sumsimdat[ind] = val;

            origindex[ind] = ind;
            ind = newind;

            val = tmpval;
        }
        sumsimdat[i] = val;
    }

    free(oldseq);
    free(newseq);
    free(hsim);
    free(origindex);
    Py_DECREF(seqs);
    Py_DECREF(weights);
    return sumsim;
}

static PyObject *
filtersim(PyObject *self, PyObject *args){
    PyArrayObject *seqs;
    int i, j, p, cutoff;
    npy_intp nseq, L;

    if(!PyArg_ParseTuple(args, "O!i", &PyArray_Type, &seqs, &cutoff)){
        return NULL;
    }

    if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
        PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
        return NULL;
    }

    if(!PyArray_ISCARRAY(seqs)){
        PyErr_SetString( PyExc_ValueError, "seq must be C-contiguous");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);

    if (nseq == 0) {
        PyErr_SetString( PyExc_ValueError, "no seqs supplied");
        return NULL;
    }

    // transpose for optimal memory order (see below)
    seqs = (PyArrayObject*)PyArray_Transpose(seqs, NULL);
    seqs = (PyArrayObject*)PyArray_Copy(seqs);
    uint8 *seqdata = PyArray_DATA(seqs);

    uint32 *hsim = malloc(sizeof(uint32)*nseq);
    for(j = 0; j < nseq; j++){
        hsim[j] = 0;
    }

    // loop over sequences, progressively cutting sequence
    // which are too similar to a focus sequence
    int N = nseq;
    for (i = 0; i < N; i++) {

        for(p = 0; p < L; p++){
            uint8 newc = seqdata[nseq*p + i];
            uint8 oldc = i == 0 ? 0xff : seqdata[nseq*p + i-1];
            uint8 *row = &seqdata[nseq*p];

            // skip rows which didn't change
            if(newc == oldc){
                continue;
            }

            // bottleneck of function, sequence transpose speeds this up
            for(j = i+1; j < N; j++){
                // using vectorizable arithmetic operations also speeds it up
                hsim[j] += (row[j] == newc) - (row[j] == oldc);
            }
            // if this needs to be _really_ fast, could use pthreads
            // (one thread per p)
        }

        // remove sequences under cutoff, moving in sequences from end
        // also find most similar sequence
        int biggestind = 0;
        uint32 biggesthsim = 0;
        for (j = i+1; j < N; j++) {
            // remove sequences by overwriting with tail sequence
            while (j < N && L-hsim[j] < cutoff) {
                for(p = 0; p < L; p++) {
                    seqdata[nseq*p + j] = seqdata[nseq*p + N-1];
                }
                hsim[j] = hsim[N-1];
                N--;
            }
            if (j < N && hsim[j] > biggesthsim) {
                biggestind = j;
                biggesthsim = hsim[j];
            }
        }

        if (i+1 >= N) {
            break;
        }

        //swap best sequence with next sequence
        for(p = 0; p < L; p++) {
            uint8 tmp = seqdata[nseq*p + biggestind];
            seqdata[nseq*p + biggestind] = seqdata[nseq*p + i+1];
            seqdata[nseq*p + i+1] = tmp;
        }
        int tmph = hsim[biggestind];
        hsim[biggestind] = hsim[i+1];
        hsim[i+1] = tmph;
    }

    free(hsim);

    npy_intp out_dims[2] = {N, L};
    npy_intp out_strides[2] = {1, nseq};
    PyObject *out = PyArray_NewFromDescr(&PyArray_Type,
                                         PyArray_DescrFromType(NPY_UINT8),
                                         2, out_dims, out_strides,
                                         seqdata, 0, NULL);
    if (out == NULL) {
        Py_DECREF(seqs);
        return NULL;
    }
    if (PyArray_SetBaseObject(out, seqs) < 0) {
        Py_DECREF(out);
        return NULL;
    }

    return (PyObject *)out;
}

/*
 * Helper function for sequence loading, which converts the sequences to
 * integer format. Takes in an array of (nseq, L+1) bytes arranged like in the
 * file (ascii + newline at the end) and overwrites the ascii with integers,
 * while also doing sanity checks. Gives speedup over numpy indexing
 * because it avoids a cast to intp.
 */
static PyObject *
translateascii(PyObject *self, PyObject *args){
    PyArrayObject *seqs;
    uint8 *seqdata;
    npy_intp nseq, L, nstride, Lstride;
    uint8 *alpha;
    unsigned int i, j, pos;
    uint8 translationtable[256];

    if(!PyArg_ParseTuple(args, "O!yi", &PyArray_Type, &seqs, &alpha, &pos)){
        return NULL;
    }

    if( (PyArray_NDIM(seqs) != 2) || PyArray_TYPE(seqs) != NPY_UINT8 ){
        PyErr_SetString( PyExc_ValueError, "seq must be 2d uint8 array");
        return NULL;
    }

    nseq = PyArray_DIM(seqs, 0);
    L = PyArray_DIM(seqs, 1);
    nstride = PyArray_STRIDE(seqs, 0);
    Lstride = PyArray_STRIDE(seqs, 1);
    seqdata = PyArray_DATA(seqs);

    for(i = 0; i < 256; i++){
        translationtable[i] = 0xff;
    }
    for(i = 0; i < strlen((char*)alpha); i++){
        if(alpha[i] >= 0xff){
            PyErr_SetString(PyExc_ValueError, "Alphabet cannot contain 0xff");
            return NULL;
        }
        translationtable[alpha[i]] = i;
    }

    for(i = 0; i < nseq; i++){
        for(j = 0; j < L; j++){

            int ind = i*nstride + j*Lstride;
            uint8 c = seqdata[ind];
            uint8 newc = translationtable[c];

            if(newc == 0xff){
                if(c == '\n'){
                    PyErr_Format(PyExc_ValueError,
                       "Sequence %d has length %d (expected %d)",
                       i+pos+1, j, (int)L);
                }
                else{
                    PyErr_Format(PyExc_ValueError,
                        "Invalid residue '%c' in sequence %d position %d",
                        c, i+pos+1, j+1);
                }
                return NULL;
            }
            seqdata[ind] = newc;
        }
    }

    Py_RETURN_NONE;
}

static PyMethodDef SeqtoolsMethods[] = {
    {"nsim", nsim, METH_VARARGS,
            "compute number of similar sequences"},
    {"nsim_weighted", nsim_weighted, METH_VARARGS,
            "compute number of similar sequences, with weights"},
    {"histsim", histsim, METH_VARARGS,
            "histogram pariwise similarities"},
    {"histsim_weighted", histsim_weighted, METH_VARARGS,
            "histogram pariwise similarities, with weights"},
    {"minsim", minsim, METH_VARARGS,
            "compute most dissimilar sequences"},
    {"sumsim", sumsim, METH_VARARGS,
            "compute sum of similarity with other all sequences"},
    {"sumsim_weighted", sumsim_weighted, METH_VARARGS,
           "compute sum of similarity with other all sequences, with weights"},
    {"filtersim", filtersim, METH_VARARGS,
            "remove sequences under a similarity cutoff to another sequence"},
    {"translateascii", translateascii, METH_VARARGS,
            "translate sequence buffer from scii to integers"},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef seqtoolsmodule = {
    PyModuleDef_HEAD_INIT,
    "seqtools",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SeqtoolsMethods
};

PyMODINIT_FUNC
PyInit_seqtools(void)
{
    import_array();
    return PyModule_Create(&seqtoolsmodule);
}
