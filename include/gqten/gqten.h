// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 14:33
* 
* Description: GraceQ/tensor project. The main header file.
*/
#ifndef GQTEN_GQTEN_H
#define GQTEN_GQTEN_H


#define GQTEN_VERSION_MAJOR 0
#define GQTEN_VERSION_MINOR 1
#define GQTEN_VERSION_PATCH 0
// GQTEN_VERSION_DEVSTR to describe the development status,
// for example the git branch name
#define GQTEN_VERSION_DEVSTR "dev-qn"


#include <string>     // string
#include <vector>     // vector
//#include <cmath>
#include <iostream>   // istream, ostream
#include <memory>     // shared_ptr

#include "gqten/detail/fwd_dcl.h"
#include "gqten/detail/consts.h"
#include "gqten/detail/value_t.h"
#include "gqten/detail/framework.h"


namespace gqten {


// Abstract base class for quantum number value
class QNVal : public Hashable, public Streamable {
public:
  QNVal(void) = default;
  virtual ~QNVal(void) = default;

  virtual QNVal *Clone(void) const = 0;

  virtual QNVal *Minus(void) const = 0;
  virtual void AddAssign(const QNVal *) = 0;


  virtual bool operator==(const QNVal &rhs) {
    return this->Hash() == rhs.Hash();
  }

  virtual bool operator!=(const QNVal &rhs) { return !(*this == rhs); }
};

using QNValPtrVec = std::vector<QNVal *>;
using QNValSharedPtr = std::shared_ptr<QNVal>;


// Abelian quantum number value
class AbelQNVal : public QNVal {
public:
  AbelQNVal(const long val) : val_(val) {}
  AbelQNVal(void) : AbelQNVal(0) {}

  AbelQNVal *Clone(void) const override { return new AbelQNVal(val_); }

  AbelQNVal *Minus(void) const override { return new AbelQNVal(-val_); }

  void AddAssign(const QNVal *prhs_b) override {
    auto prhs_d = static_cast<const AbelQNVal *>(prhs_b);   // Do safe downcasting
    val_ += prhs_d->val_;
  }

  // Override for Hashable base class
  std::size_t Hash(void) const override { return hasher_(val_); }

  // Override for Streamable base class
  void StreamRead(std::istream &is) override { is >> val_; }
  void StreamWrite(std::ostream &os) const override { os << val_ << std::endl; }

private:
  long val_;
  std::hash<long> hasher_;
};

using NormalQNVal = AbelQNVal;


// Quantum number card for name-value pair
class QNCard {
public:
  QNCard(const std::string &name, const QNVal &val) :
      name_(name), pval_(val.Clone()) {}

  QNValSharedPtr GetValPtr(void) const { return pval_; }

private:
  std::string name_;
  QNValSharedPtr pval_;
};

using QNCardVec = std::vector<QNCard>;


// Quantum number class
template <typename... QNValTs>
class QN : public Hashable, public Streamable {
public:
  QN(void);
  QN(const QNCardVec &);
  ~QN(void);

  QN(const QN &);
  QN &operator=(const QN &);

  QN operator-(void) const;
  QN &operator+=(const QN &);

  QN operator+(const QN &rhs) const {
    QN sum(*this);
    sum += rhs;
    return sum;
  }
  QN operator-(const QN &rhs) const { return (*this) + (-rhs); }

  bool operator==(const QN &rhs) const { return hash_ == rhs.hash_; }
  bool operator!=(const QN &rhs) const { return !(*this == rhs); }

  std::size_t Hash(void) const override { return hash_; }

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

private:
  QNValPtrVec qnvals_;
  std::size_t hash_;

  std::size_t CalcHash(void) const;

  QN(const QNValPtrVec &);    // Intra used private constructor
};


// Quantum number sector.
template <typename QNT>
class QNSector : public Hashable, public Streamable {
public:
  QNSector(const QNT &qn, const QNSectorDimT_ dim) : qn(qn), dim(dim) {
    hash_ = CalcHash();
  }
  QNSector(void) : QNSector(QNT(), 0) {}

  QNSector(const QNSector &qnsct) :
      qn(qnsct.qn), dim(qnsct.dim), hash_(qnsct.hash_) {}
  QNSector &operator=(const QNSector &rhs) {
    qn = rhs.qn;
    dim = rhs.dim;
    hash_ = rhs.hash_;
    return *this;
  }

  bool operator==(const QNSector &rhs) const { return hash_ == rhs.hash_; }
  bool operator!=(const QNSector &rhs) const { return !(*this == rhs); }

  std::size_t Hash(void) const override { return hash_; }

  void StreamRead(std::istream &is) override {
    is >> qn >> dim >> hash_;
  }

  void StreamWrite(std::ostream &os) const override {
    os << qn;
    os << dim << std::endl;
    os << hash_ << std::endl;
  }

  QNT qn;
  QNSectorDimT_ dim;

private:
  std::size_t CalcHash(void) const { return qn.Hash() ^ dim; }
  std::size_t hash_;
};

template <typename QNT>
using QNSectorVec = std::vector<QNSector<QNT>>;

template <typename QNT>
using ConstQNSectorPtrVec = std::vector<const QNSector<QNT> *>;


// Quantum number sector set.
template <typename QNT>
class QNSectorSet : public Hashable {   // Only hashable here
public:
  QNSectorSet(void) {}
  QNSectorSet(const QNSectorVec<QNT> &qnscts) : qnscts(qnscts) {}
  QNSectorSet(const ConstQNSectorPtrVec<QNT> &pqnscts) {
    for (auto &pqnsct : pqnscts) { qnscts.push_back(*pqnsct); }
  }
  virtual ~QNSectorSet() = default;

  QNSectorSet(const QNSectorSet &qnss) : qnscts(qnss.qnscts) {}

  bool operator==(const QNSectorSet &rhs) const { return Hash() == rhs.Hash(); }
  bool operator!=(const QNSectorSet &rhs) const { return !(*this == rhs); }

  std::size_t Hash(void) const override { return VecHasher(qnscts); }

  QNSectorVec<QNT> qnscts;
};


template <typename QNT>
bool operator==(const QNSectorVec<QNT> &lhs, const QNSectorSet<QNT> &rhs) {
  return VecHasher(lhs) == rhs.Hash();
}


template <typename QNT>
bool operator!=(const QNSectorVec<QNT> &lhs, const QNSectorSet<QNT> &rhs) {
  return !(lhs == rhs);
}


// Index.
template <typename QNT>
struct InterOffsetQnsct {
  InterOffsetQnsct(const IndexDimT_ &inter_offset, const QNSector<QNT> &qnsct) :
      inter_offset(inter_offset), qnsct(qnsct) {}
  IndexDimT_ inter_offset;
  QNSector<QNT> qnsct;
};


/* TODO: name change: Index -> TenIdx */
template <typename QNT>
class Index : public QNSectorSet<QNT>, public Streamable {
public:
  using QNSectorVecT = QNSectorVec<QNT>;
  using QNSctSetT = QNSectorSet<QNT>;

  Index(void) : QNSctSetT(), dim(0), dir(NDIR), tag(kEmptyStr_) {}
  Index(
      const QNSectorVecT &qnscts,
      const std::string &dir,
      const std::string &tag) : QNSctSetT(qnscts), dir(dir), tag(tag) {
      dim = CalcDim();
  }
  Index(const QNSectorVecT &qnscts) : Index(qnscts, NDIR, kEmptyStr_) {}
  Index(const QNSectorVecT &qnscts, const std::string &dir) :
      Index(qnscts, dir, kEmptyStr_) {}

  Index(const Index &index) :
      Index(index.qnscts, index.dim, index.dir, index.tag) {}
  Index &operator=(const Index &rhs) {
    QNSctSetT::qnscts = rhs.qnscts;
    dim = rhs.dim;
    dir = rhs.dir;
    tag = rhs.tag;
    return *this;
  }

  IndexDimT_ CalcDim(void) {
    IndexDimT_ dim = 0;
    for (auto &qnsct : QNSctSetT::qnscts) {
      dim += qnsct.dim;
    }
    return dim;
  }

  bool operator==(const Index &rhs) const { return  Hash() ==  rhs.Hash(); }

  InterOffsetQnsct<QNT> CoorInterOffsetAndQnsct(const long) const;

  // Inplace inverse operation
  /* TODO: name change: Dag() -> ?? */
  void Dag(void) {
    if (dir == IN) {
      dir = OUT;
    } else if (dir == OUT) {
      dir = IN;
    }
  }

  std::size_t Hash(void) const override {
    return QNSctSetT::Hash() ^ str_hasher_(tag);
  }

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  IndexDimT_ dim;
  std::string dir;
  std::string tag;

private:
  Index(
      const QNSectorVecT &qnscts,
      const IndexDimT_ dim,
      const std::string &dir,
      const std::string &tag
  ) : QNSctSetT(qnscts), dim(dim), dir(dir), tag(tag) {}

  std::hash<std::string> str_hasher_;
};


template <typename IndexT>
IndexT InverseIndex(const IndexT &);


// Dense block labeled by the quantum number.
/* TODO: name change: ElemType -> ElemT */
/* TODO: refactor: QNBlock does not manage the raw data
 * but just point data's location. Raw data will be managed by GQTensor.
 */
template <typename QNT, typename ElemType>
std::vector<QNBlock<QNT, ElemType> *> BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const ElemType,
    const std::vector<QNBlock<QNT, ElemType> *> &,
    const std::vector<QNBlock<QNT, ElemType> *> &
);

template <typename QNT, typename ElemType>
class QNBlock : public QNSectorSet<QNT>, public Streamable {
// Some functions called by tensor numerical functions to use the private constructor.
friend std::vector<QNBlock<QNT, ElemType> *> BlocksCtrctBatch<QNT, ElemType>(
    const std::vector<long> &, const std::vector<long> &,
    const ElemType,
    const std::vector<QNBlock<QNT, ElemType> *> &,
    const std::vector<QNBlock<QNT, ElemType> *> &
);

public:
  using QNSctVecT = QNSectorVec<QNT>;
  using QNSctSetT = QNSectorSet<QNT>;

  QNBlock(void) = default;
  QNBlock(const QNSctVecT &);

  QNBlock(const QNBlock &);
  QNBlock &operator=(const QNBlock &);
  
  QNBlock(QNBlock &&) noexcept;
  QNBlock &operator=(QNBlock &&) noexcept;

  ~QNBlock(void) override;

  // Element getter and setter.
  const ElemType &operator()(const std::vector<long> &) const;
  ElemType &operator()(const std::vector<long> &);

  // Data access.
  const ElemType *cdata(void) const { return data_; }   // constant reference.
  ElemType * &data(void) { return data_; }              // non-constant reference.

  // Hash methods.
  std::size_t PartHash(const std::vector<long> &) const;
  /* TODO: change name: QNSectorSetHash -> QNSctSetHash */
  std::size_t QNSectorSetHash(void) const { return qnscts_hash_; }

  // Inplace operations.
  void Random(void);
  void Transpose(const std::vector<long> &);

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  // Public data members.
  long ndim = 0;              // Number of dimensions.
  std::vector<long> shape;    // Shape of the block.
  long size = 0;              // Total number of elements in this block.

private:
  // NOTE: For performance reason, this constructor will NOT initialize
  // the data_ to 0!!! So, it should only be intra-used.
  QNBlock(const ConstQNSectorPtrVec<QNT> &);

  ElemType *data_ = nullptr;    // Data in a 1D array.
  std::vector<long> data_offsets_;
  std::size_t qnscts_hash_ = 0;
};


// GQTensor
template <typename QNT>
struct BlkInterOffsetsAndQNSS {     // QNSS: QNSectorSet.
  BlkInterOffsetsAndQNSS(
      const std::vector<long> &blk_inter_offsets,
      const QNSectorSet<QNT> &blk_qnss
  ) : blk_inter_offsets(blk_inter_offsets), blk_qnss(blk_qnss) {}

  std::vector<long> blk_inter_offsets;
  QNSectorSet<QNT> blk_qnss;
};


/* TODO: name change: ElemType -> ElemT */
template <typename QNT, typename ElemType>
class GQTensor : public Streamable {
public:
  using IdxVecT = std::vector<Index<QNT>>;

  GQTensor(void) = default;
  GQTensor(const std::vector<Index<QNT>> &);

  GQTensor(const GQTensor &);
  GQTensor &operator=(const GQTensor &);

  GQTensor(GQTensor &&) noexcept;
  GQTensor &operator=(GQTensor &&) noexcept;

  ~GQTensor(void);

  // Element getter and setter.
  ElemType Elem(const std::vector<long> &) const;     // Getter.
  ElemType &operator()(const std::vector<long> &);    // Setter.

  // Access to the blocks.
  const std::vector<QNBlock<QNT, ElemType> *> &cblocks(void) const {
    return blocks_;
  }
  std::vector<QNBlock<QNT, ElemType> *> &blocks(void) { return blocks_; }

  // Inplace operations.

  // Random set tensor elements with given quantum number divergence.
  // Any original blocks will be destroyed.
  void Random(const QNT &);

  // Tensor transpose.
  void Transpose(const std::vector<long> &);

  // Normalize the GQTensor and return its norm.
  GQTEN_Double Normalize(void);

  // Switch the direction of the indexes, complex conjugate of the element.
  void Dag(void);

  // Operators overload.
  GQTensor operator-(void) const;
  GQTensor operator+(const GQTensor &);
  GQTensor &operator+=(const GQTensor &);

  bool operator==(const GQTensor &) const;
  bool operator!=(const GQTensor &rhs) const { return !(*this == rhs); }

  // Iterators.
  // Return all the tensor coordinates. So heavy that you should not use it!
  std::vector<std::vector<long>> CoorsIter(void) const;

  void StreamRead(std::istream &) override;
  void StreamWrite(std::ostream &) const override;

  // Public data members.
  IdxVecT indexes;
  ElemType scalar = 0.0;
  std::vector<long> shape;

private:
  std::vector<QNBlock<QNT, ElemType> *> blocks_;

  double Norm(void);

  BlkInterOffsetsAndQNSS<QNT> CalcTargetBlkInterOffsetsAndQNSS(
      const std::vector<long> &) const;
  std::vector<QNSectorSet<QNT>> BlkQNSSsIter(void) const;

};

// Tensor type with real/complex number element
template <typename QNT>
using RealGQTensor = GQTensor<QNT, GQTEN_Double>;

template <typename QNT>
using CplxGQTensor = GQTensor<QNT, GQTEN_Complex>;

// Some tensor operations
template <typename GQTensorT>
GQTensorT Dag(const GQTensorT &);

// Just mock the dag. Not construct a new object.
template <typename GQTensorT>
inline const GQTensorT &MockDag(const GQTensorT &t) { return t; }

template <typename QNT, typename ElemType>
QNT Div(const GQTensor<QNT, ElemType> &);

template <typename QNT, typename ElemType>
GQTensor<QNT, ElemType> operator*(
    const GQTensor<QNT, ElemType> &, const ElemType &
);

template <typename QNT, typename ElemType>
GQTensor<QNT, ElemType> operator*(
    const ElemType &, const GQTensor<QNT, ElemType> &
);

template <typename QNT>
CplxGQTensor<QNT> ToComplex(const RealGQTensor<QNT> &);


// Tensor numerical functions.
// Tensors contraction.
template <typename QNT, typename TenElemType>
void Contract(
    const GQTensor<QNT, TenElemType> *, const GQTensor<QNT, TenElemType> *,
    const std::vector<std::vector<long>> &,
    GQTensor<QNT, TenElemType> *);

// These APIs just for forward compatibility, it will be deleted soon.
// TODO: Remove these API.
template <typename QNT>
inline RealGQTensor<QNT> *Contract(
    const RealGQTensor<QNT> &ta, const RealGQTensor<QNT> &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new RealGQTensor<QNT>();
  Contract(&ta, &tb, axes_set, res_t);
  return res_t;
}

template <typename QNT>
inline CplxGQTensor<QNT> *Contract(
    const CplxGQTensor<QNT> &ta, const CplxGQTensor<QNT> &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new CplxGQTensor<QNT>();
  Contract(&ta, &tb, axes_set, res_t);
  return res_t;
}

template <typename QNT>
inline CplxGQTensor<QNT> *Contract(
    const RealGQTensor<QNT> &ta, const CplxGQTensor<QNT> &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new CplxGQTensor<QNT>();
  auto zta = ToComplex(ta);
  Contract(&zta, &tb, axes_set, res_t);
  return res_t;
}

template <typename QNT>
inline CplxGQTensor<QNT> *Contract(
    const CplxGQTensor<QNT> &ta, const RealGQTensor<QNT> &tb,
    const std::vector<std::vector<long>> &axes_set) {
  auto res_t = new CplxGQTensor<QNT>();
  auto ztb = ToComplex(tb);
  Contract(&ta, &ztb, axes_set, res_t);
  return res_t;
}


//// Tensors linear combination.
//// Do the operation: res += (coefs[0]*ts[0] + coefs[1]*ts[1] + ...).
////[> TODO: Support scalar (rank 0) tensor case. <]
//template <typename TenElemType>
//void LinearCombine(
    //const std::vector<TenElemType> &,
    //const std::vector<GQTensor<TenElemType> *> &,
    //GQTensor<TenElemType> *);

//template <typename TenElemType>
//void LinearCombine(
    //const std::size_t,
    //const TenElemType *,
    //const std::vector<GQTensor<TenElemType> *> &,
    //GQTensor<TenElemType> *);

//inline void LinearCombine(
    //const std::size_t size,
    //const double *dcoefs,
    //const std::vector<GQTensor<GQTEN_Complex> *> &zts,
    //GQTensor<GQTEN_Complex> *res) {
  //auto zcoefs = new GQTEN_Complex [size];
  //for (size_t i = 0; i < size; ++i) {
    //zcoefs[i] = dcoefs[i];
  //}
  //LinearCombine(size, zcoefs, zts, res);
  //delete [] zcoefs;
//}


//// Tensor SVD.
//template <typename TenElemType>
//void Svd(
    //const GQTensor<TenElemType> *,
    //const long, const long,
    //const QN &, const QN &,
    //const double, const long, const long,
    //GQTensor<TenElemType> *,
    //GQTensor<GQTEN_Double> *,
    //GQTensor<TenElemType> *,
    //double *, long *);


//// These APIs just for forward compatibility, it will be deleted soon.
//// TODO: Remove these APIs.
//template <typename TenElemType>
//struct SvdRes {
  //SvdRes(
      //GQTensor<TenElemType> *u,
      //GQTensor<GQTEN_Double> *s,
      //GQTensor<TenElemType> *v,
      //const double trunc_err, const long D) :
      //u(u), s(s), v(v), trunc_err(trunc_err), D(D) {}
  //GQTensor<TenElemType> *u;
  //GQTensor<GQTEN_Double> *s;
  //GQTensor<TenElemType> *v;
  //const double trunc_err;
  //const long D;
//};


//template <typename TenElemType>
//inline SvdRes<TenElemType> Svd(
    //const GQTensor<TenElemType> &t,
    //const long ldims, const long rdims,
    //const QN &ldiv, const QN &rdiv,
    //const double cutoff, const long Dmin, const long Dmax) {
  //auto pu =  new GQTensor<TenElemType>();
  //auto ps =  new GQTensor<GQTEN_Double>();
  //auto pvt = new GQTensor<TenElemType>();
  //double trunc_err;
  //long D;
  //Svd(
      //&t,
      //ldims, rdims,
      //ldiv, rdiv,
      //cutoff, Dmin, Dmax,
      //pu, ps, pvt,
      //&trunc_err, &D);
  //return SvdRes<TenElemType>(pu, ps, pvt,trunc_err, D);
//}


//template <typename TenElemType>
//inline SvdRes<TenElemType> Svd(
    //const GQTensor<TenElemType> &t,
    //const long ldims, const long rdims,
    //const QN &ldiv, const QN &rdiv) {
  //auto t_shape = t.shape;
  //long lsize = 1;
  //long rsize = 1;
  //for (std::size_t i = 0; i < t_shape.size(); ++i) {
    //if (i < ldims) {
      //lsize *= t_shape[i];
    //} else {
      //rsize *= t_shape[i];
    //}
  //}
  //auto D = ((lsize >= rsize) ? lsize : rsize);
  //return Svd(
      //t,
      //ldims, rdims,
      //ldiv, rdiv,
      //0, D, D);
//}


// Tensor transpose function multi-thread controller.
int GQTenGetTensorTransposeNumThreads(void);

void GQTenSetTensorTransposeNumThreads(const int);


// Timer.
class Timer {
public:
  Timer(const std::string &);

  void Restart(void);
  double Elapsed(void);
  double PrintElapsed(std::size_t precision = 5);

private:
  double start_;
  std::string notes_;

  double GetWallTime(void);
};
} /* gqten */ 


// Include implementation details.
#include "gqten/detail/qn_impl.h"
#include "gqten/detail/index_impl.h"
#include "gqten/detail/qnblock_impl.h"
#include "gqten/detail/gqtensor_impl.h"
#include "gqten/detail/ten_ctrct_impl.h"
//#include "gqten/detail/ten_lincmb_impl.h"
//#include "gqten/detail/ten_svd_impl.h"


#endif /* ifndef GQTEN_GQTEN_H */
