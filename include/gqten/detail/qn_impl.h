// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-02-26 16:04
* 
* Description: GraceQ/tensor project. Implementation details for quantum number class template.
*/
#include "gqten/gqten.h"
#include "gqten/detail/vec_hash.h"

#include <iostream>     // istream, ostream

#include <assert.h>     // assert


#ifdef Release
  #define NDEBUG
#endif


namespace gqten {


template <typename... QNValTs>
inline std::size_t QN<QNValTs...>::CalcHash(void) const {
  if (qnvals_.size() == 0) {
    return 0; 
  } else {
    return VecPtrHasher(qnvals_);
  }
}


template <typename... QNValTs>
QN<QNValTs...>::QN(void) { hash_ = CalcHash(); }


template <typename... QNValTs>
QN<QNValTs...>::QN(const QNCardVec &qncards) {
  for (auto &qncard : qncards) {
    qnvals_.push_back((qncard.GetValPtr())->Clone());
  }
  hash_ = CalcHash();
}


// WARNING: private constructor
template <typename... QNValTs>
QN<QNValTs...>::QN(const QNValPtrVec &qnvals) {
  qnvals_ = qnvals;
  hash_ = CalcHash();
}


template <typename... QNValTs>
QN<QNValTs...>::~QN(void) {
  for (auto &qnval : qnvals_) {
    delete qnval;
  }
}


template <typename... QNValTs>
QN<QNValTs...>::QN(const QN &qn) :
    hash_(qn.hash_) {
  for (auto &qnval : qn.qnvals_) {
    qnvals_.push_back(qnval->Clone());
  }
}


template <typename... QNValTs>
QN<QNValTs...> &QN<QNValTs...>::operator=(const QN &rhs) {
  for (auto &qnval : qnvals_) {
    delete qnval;
  }
  qnvals_.clear();
  for (auto &qnval : rhs.qnvals_) {
    qnvals_.push_back(qnval->Clone());
  }
  hash_ = rhs.hash_;
  return *this;
}


template <typename... QNValTs>
QN<QNValTs...> QN<QNValTs...>::operator-(void) const {
  QNValPtrVec new_qnvals;
  for (auto &qnval : qnvals_) {
    new_qnvals.push_back(qnval->Minus());
  }
  return QN(new_qnvals);    // WARNING: use private constructor here
}


template <typename... QNValTs>
QN<QNValTs...> &QN<QNValTs...>::operator+=(const QN &rhs) {
  auto qnvals_size = this->qnvals_.size();
  assert(qnvals_size == rhs.qnvals_.size());
  for (std::size_t i = 0; i < qnvals_size; ++i) {
    (this->qnvals_[i])->AddAssign(rhs.qnvals_[i]);
  }
  hash_ = CalcHash();
  return *this;
}


template <typename... QNValTs>
void QN<QNValTs...>::StreamRead(std::istream &is) {
  qnvals_ = {(new QNValTs)...};     // Initialize the quantum number value slots
  for (auto &qnval : qnvals_) {
    is >> (*qnval);
  }
  is >> hash_;
}


template <typename... QNValTs>
void QN<QNValTs...>::StreamWrite(std::ostream &os) const {
  for (auto &qnval : qnvals_) {
    os << (*qnval) << std::endl;
  }
  os << hash_ << std::endl;
}
} /* gqten */ 
