// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-11 15:24
* 
* Description: GraceQ/tensor project. Implementation details for quantum number block class template.
*/
#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>

#include <assert.h>

#include "gqten/gqten.h"
#include "gqten/detail/vec_hash.h"
#include "gqten/detail/utils_inl.h"


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


// Forward declarations.
std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);


GQTEN_Double *DenseTensorTranspose(
    const GQTEN_Double *,
    const long,
    const long,
    const std::vector<long> &,
    const std::vector<long> &);


GQTEN_Complex *DenseTensorTranspose(
    const GQTEN_Complex *,
    const long,
    const long,
    const std::vector<long> &,
    const std::vector<long> &);


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType>::QNBlock(const QNSctVecT &init_qnscts) :
    QNSctSetT(init_qnscts) {
  ndim = QNSctSetT::qnscts.size(); 
  for (auto &qnsct : QNSctSetT::qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) { size *= shape[i]; }
    data_ = (ElemType *)calloc(size, sizeof(ElemType));    // Allocate memory and initialize to 0.
    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSctSetT::Hash();
  }
}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType>::QNBlock(const ConstQNSectorPtrVec<QNT> &pinit_qnscts) :
    QNSctSetT(pinit_qnscts) {

#ifdef GQTEN_TIMING_MODE
    Timer qnblk_intra_construct_timer("qnblk_intra_construct");
    qnblk_intra_construct_timer.Restart();
#endif

  ndim = QNSctSetT::qnscts.size(); 
  for (auto &qnsct : QNSctSetT::qnscts) {
    shape.push_back(qnsct.dim);
  }
  if (ndim != 0) {
    size = 1;       // Initialize the block size.
    for (long i = 0; i < ndim; ++i) {
      size *= shape[i];
    }

#ifdef GQTEN_TIMING_MODE
    Timer qnblk_intra_construct_new_data_timer("qnblk_intra_construct_new_data");
    qnblk_intra_construct_new_data_timer.Restart();
#endif

    data_ = (ElemType *)malloc(size * sizeof(ElemType));      // Allocate memory. NOT INITIALIZE TO ZERO!!!

#ifdef GQTEN_TIMING_MODE
    qnblk_intra_construct_new_data_timer.PrintElapsed(8);
#endif

    data_offsets_ = CalcMultiDimDataOffsets(shape);
    qnscts_hash_ = QNSctSetT::Hash();
  }

#ifdef GQTEN_TIMING_MODE
    qnblk_intra_construct_timer.PrintElapsed(8);
#endif

}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType>::QNBlock(const QNBlock &qnblk) :
    QNSctSetT(qnblk),   // Use copy constructor of the base class.
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_) {
  data_ = (ElemType *)malloc(size * sizeof(ElemType));
  std::memcpy(data_, qnblk.data_, size * sizeof(ElemType));
}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType> &QNBlock<QNT, ElemType>::operator=(const QNBlock &rhs) {
  // Copy data.
  auto new_data = (ElemType *)malloc(rhs.size * sizeof(ElemType));
  std::memcpy(new_data, rhs.data_, rhs.size * sizeof(ElemType));
  free(data_);
  data_ = new_data;
  // Copy other members.
  QNSctSetT::qnscts = rhs.qnscts;    // For the base class.
  ndim = rhs.ndim;
  shape = rhs.shape;
  size = rhs.size;
  data_offsets_ = rhs.data_offsets_;
  qnscts_hash_ = rhs.qnscts_hash_;
  return *this;
}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType>::QNBlock(QNBlock &&qnblk) noexcept :
    QNSctSetT(qnblk),
    ndim(qnblk.ndim),
    shape(qnblk.shape),
    size(qnblk.size),
    data_offsets_(qnblk.data_offsets_),
    qnscts_hash_(qnblk.qnscts_hash_),
    data_(qnblk.data_) {
  qnblk.data_ = nullptr;
}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType> &
QNBlock<QNT, ElemType>::operator=(QNBlock &&rhs) noexcept {
  // Move data.
  free(data_);
  data_ = rhs.data_;
  rhs.data_ = nullptr;
  // Copy other members.
  QNSctSetT::qnscts = rhs.qnscts;    // For the base class.
  ndim = rhs.ndim;
  shape = rhs.shape;
  size = rhs.size;
  data_offsets_ = rhs.data_offsets_;
  qnscts_hash_ = rhs.qnscts_hash_;
  return *this;
}


template <typename QNT, typename ElemType>
QNBlock<QNT, ElemType>::~QNBlock(void) {
  free(data_);
  data_ = nullptr;
}


// Block element getter.
template <typename QNT, typename ElemType>
const ElemType &QNBlock<QNT, ElemType>::operator()(
    const std::vector<long> &coors) const {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


// Block element setter.
template <typename QNT, typename ElemType>
ElemType &QNBlock<QNT, ElemType>::operator()(const std::vector<long> &coors) {
  assert(coors.size() == ndim);
  auto offset = CalcEffOneDimArrayOffset(coors, ndim, data_offsets_);
  return *(data_+offset);
}


template <typename QNT, typename ElemType>
std::size_t QNBlock<QNT, ElemType>::PartHash(
    const std::vector<long> &axes
) const {
  auto selected_qnscts_ndim  = axes.size();
  ConstQNSectorPtrVec<QNT> pselected_qnscts(selected_qnscts_ndim);
  for (std::size_t i = 0; i < selected_qnscts_ndim; ++i) {
    pselected_qnscts[i] = &QNSctSetT::qnscts[axes[i]];
  }
  return VecPtrHasher(pselected_qnscts);
}


// Inplace operation.
template <typename QNT, typename ElemType>
void QNBlock<QNT, ElemType>::Random(void) {
  for (int i = 0; i < size; ++i) { Rand(data_[i]); }
}


template <typename QNT, typename ElemType>
void QNBlock<QNT, ElemType>::Transpose(
    const std::vector<long> &transed_axes
) {
  QNSctVecT transed_qnscts(ndim);
  std::vector<long> transed_shape(ndim);
  for (long i = 0; i < ndim; ++i) {
    transed_qnscts[i] = QNSctSetT::qnscts[transed_axes[i]];
    transed_shape[i] = transed_qnscts[i].dim;
  }
  auto transed_data_offsets_ = CalcMultiDimDataOffsets(transed_shape);
  auto new_data = DenseTensorTranspose(
                      data_,
                      ndim, size, shape,
                      transed_axes);
  free(data_);
  data_ = new_data;
  shape = transed_shape;
  QNSctSetT::qnscts = transed_qnscts;
  data_offsets_ = transed_data_offsets_;
}


template <typename QNT, typename ElemType>
void QNBlock<QNT, ElemType>::StreamRead(std::istream &is) {
  is >> ndim;
  
  is >> size;

  shape = std::vector<long>(ndim);
  for (auto &order : shape) { is >> order; }

  QNSctSetT::qnscts = QNSctVecT(ndim);
  for (auto &qnsct : QNSctSetT::qnscts) { is >> qnsct; }

  data_offsets_ = std::vector<long>(ndim);
  for (auto &offset : data_offsets_) { is >> offset; }

  is >> qnscts_hash_;

  is.seekg(1, std::ios::cur);    // Skip the line break.

  if (size != 0) {
    data_ = (ElemType *)malloc(size * sizeof(ElemType));
    is.read((char *) data_, size*sizeof(ElemType));
  }
}


template <typename QNT, typename ElemType>
void QNBlock<QNT, ElemType>::StreamWrite(std::ostream &os) const {
  os << ndim << std::endl;

  os << size << std::endl;

  for (auto &order : shape) { os << order << std::endl; }

  for (auto &qnsct : QNSctSetT::qnscts) { os << qnsct; }

  for (auto &offset : data_offsets_) { os << offset << std::endl; }

  os << qnscts_hash_ << std::endl;

  if (size != 0) {
    os.write((char *) data_, size*sizeof(ElemType));
  }
  os << std::endl;
}
} /* gqten */ 
