// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 16:37
* 
* Description: GraceQ/tensor project. Implementation details about utility classes and functions.
*/
#include "utils.h"

#include <vector>

#include <assert.h>

#include "gqten/gqten.h"

#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


//GQTensor<GQTEN_Complex> ToComplex(const GQTensor<GQTEN_Double> &dten) {
  //GQTensor<GQTEN_Complex> zten(dten.indexes);
  //if (dten.scalar != 0.0) {
    //zten.scalar = dten.scalar;
    //return zten;
  //}
  //for (auto &pdtenblk : dten.cblocks()) {
    //auto pztenblk = new QNBlock<GQTEN_Complex>(pdtenblk->qnscts);
    //assert(pztenblk->size == pdtenblk->size);
    //ArrayToComplex(pztenblk->data(), pdtenblk->cdata(), pdtenblk->size);
    //zten.blocks().push_back(pztenblk);
  //}
  //return zten;
//}


//QN CalcDiv(
    //const std::vector<QNSector> &qnscts, const std::vector<Index> &indexes) {
  //QN div;
  //auto ndim = indexes.size();
  //assert(qnscts.size() == ndim);
  //for (size_t i = 0; i < ndim; ++i) {
    //if (indexes[i].dir == IN) {
      //auto qnflow = -qnscts[i].qn;
      //if (ndim == 1) {
        //return qnflow;
      //} else {
        //if (i == 0) {
          //div = qnflow;
        //} else {
          //div += qnflow;
        //}
      //}
    //} else if (indexes[i].dir == OUT) {
      //auto qnflow = qnscts[i].qn;
      //if (ndim == 1) {
        //return qnflow;
      //} else {
        //if (i == 0) {
          //div = qnflow;
        //} else {
          //div += qnflow;
        //}
      //}
    //} 
  //}
  //return div;
//}


//QN CalcDiv(const QNSectorSet &blk_qnss, const std::vector<Index> &indexes) {
  //return CalcDiv(blk_qnss.qnscts, indexes);
//}


// Multiplication from vec[i] to the end.
long MulToEnd(const std::vector<long> &v, int i) {
  assert(i < v.size());
  long mul = 1;
  for (auto it = v.begin()+i; it != v.end(); ++it) {
    mul *= *it; 
  } 
  return mul;
}


std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &shape) {
  auto ndim = shape.size();
  std::vector<long> offsets(ndim);
  for (size_t i = 0; i < ndim; ++i) {
    if (i == ndim-1) {
      offsets[i] = 1;
    } else {
      offsets[i] = MulToEnd(shape, i+1);
    }
  }
  return offsets;
}


std::vector<std::vector<long>> GenAllCoors(const std::vector<long> &shape) {
  std::vector<std::vector<long>> each_coors(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    for (long j = 0; j < shape[i]; ++j) {
      each_coors[i].push_back(j);
    }
  }
  return CalcCartProd(each_coors);
}
} /* gqten */ 
