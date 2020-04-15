// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-10 19:39
* 
* Description: GraceQ/tensor project. Unittests for tensor contraction functions.
*/
#include <cmath>

#include "gtest/gtest.h"

#include "gqten/gqten.h"
#include "testing_utils.h"

#include "mkl.h"    // Included after other header file. Because GraceQ needs redefine MKL_Complex16 to gqten::GQTEN_Complex .


using namespace gqten;


using NormalQN1 = QN<NormalQNVal>;
using QNSctT = QNSector<NormalQN1>;
using IndexT = Index<NormalQN1>;
using DGQTenT = GQTensor<NormalQN1, GQTEN_Double>;
using ZGQTenT = GQTensor<NormalQN1, GQTEN_Complex>;


struct TestContraction : public testing::Test {
  std::string qn_nm = "qn";
  NormalQN1 qn0 =  NormalQN1({QNCard(qn_nm, NormalQNVal( 0))});
  NormalQN1 qnp1 = NormalQN1({QNCard(qn_nm, NormalQNVal( 1))});
  NormalQN1 qnp2 = NormalQN1({QNCard(qn_nm, NormalQNVal( 2))});
  NormalQN1 qnm1 = NormalQN1({QNCard(qn_nm, NormalQNVal(-1))});
  int d_s = 3;
  QNSctT qnsct0_s =  QNSctT(qn0,  d_s);
  QNSctT qnsctp1_s = QNSctT(qnp1, d_s);
  QNSctT qnsctm1_s = QNSctT(qnm1, d_s);
  int d_l = 10;
  QNSctT qnsct0_l =  QNSctT(qn0,  d_l);
  QNSctT qnsctp1_l = QNSctT(qnp1, d_l);
  QNSctT qnsctm1_l = QNSctT(qnm1, d_l);
  IndexT idx_in_s =  IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, IN);
  IndexT idx_out_s = IndexT({qnsctm1_s, qnsct0_s, qnsctp1_s}, OUT);
  IndexT idx_in_l =  IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, IN);
  IndexT idx_out_l = IndexT({qnsctm1_l, qnsct0_l, qnsctp1_l}, OUT);

  DGQTenT dten_1d_s = DGQTenT({idx_out_s});
  DGQTenT dten_1d_l = DGQTenT({idx_out_l});
  DGQTenT dten_2d_s = DGQTenT({idx_in_s, idx_out_s});
  DGQTenT dten_2d_l = DGQTenT({idx_in_l, idx_out_l});
  DGQTenT dten_3d_s = DGQTenT({idx_in_s, idx_out_s, idx_out_s});
  DGQTenT dten_3d_l = DGQTenT({idx_in_l, idx_out_l, idx_out_l});

  ZGQTenT zten_1d_s = ZGQTenT({idx_out_s});
  ZGQTenT zten_1d_l = ZGQTenT({idx_out_l});
  ZGQTenT zten_2d_s = ZGQTenT({idx_in_s, idx_out_s});
  ZGQTenT zten_2d_l = ZGQTenT({idx_in_l, idx_out_l});
  ZGQTenT zten_3d_s = ZGQTenT({idx_in_s, idx_out_s, idx_out_s});
  ZGQTenT zten_3d_l = ZGQTenT({idx_in_l, idx_out_l, idx_out_l});
};


template <typename QNT, typename TenElemType>
void RunTestTenCtrct1DCase(GQTensor<QNT, TenElemType> &t, const QNT &div) {
  t.Random(div);
  TenElemType res = 0;
  for (auto &blk : t.cblocks()) {
    for (long i = 0; i < blk->size; ++i) {
      res += std::norm(blk->cdata()[i]);
    }
  }
  GQTensor<QNT, TenElemType> t_res;
  auto t_dag = Dag(t);
  Contract(&t, &t_dag, {{0}, {0}}, &t_res);
  GtestExpectNear(t_res.scalar, res, kEpsilon);
}


TEST_F(TestContraction, 1DCase) {
  RunTestTenCtrct1DCase(dten_1d_s, qn0);
  RunTestTenCtrct1DCase(dten_1d_s, qnp1);
  RunTestTenCtrct1DCase(dten_1d_s, qnm1);

  RunTestTenCtrct1DCase(zten_1d_s, qn0);
  RunTestTenCtrct1DCase(zten_1d_s, qnp1);
  RunTestTenCtrct1DCase(zten_1d_s, qnm1);
}


template <typename QNT, typename TenElemType>
void RunTestTenCtrct2DCase1(
    GQTensor<QNT, TenElemType> &ta, GQTensor<QNT, TenElemType> &tb
) {
  auto m = ta.shape[0];
  auto n = tb.shape[1];
  auto k1 = ta.shape[1];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n
  );
  GQTensor<QNT, TenElemType> res;
  Contract(&ta, &tb, {{1}, {0}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    GtestExpectNear(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename QNT, typename TenElemType>
void RunTestTenCtrct2DCase2(
    GQTensor<QNT, TenElemType> &ta, GQTensor<QNT, TenElemType> &tb
) {
  auto m = ta.shape[0];
  auto n = tb.shape[1];
  auto k1 = ta.shape[1];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem({coor[1], coor[0]});
    idx++;
  }
  TenElemType res_scalar = 0.0;
  for (long i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<QNT, TenElemType> res;
  Contract(&ta, &tb, {{0, 1}, {1, 0}}, &res);
  GtestExpectNear(res.scalar, res_scalar, kEpsilon);
  delete [] dense_ta;
  delete [] dense_tb;
}


TEST_F(TestContraction, 2DCase) {
  auto dten_2d_s2 = dten_2d_s;
  dten_2d_s.Random(qn0);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);
  dten_2d_s.Random(qnp1);
  dten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DCase1(dten_2d_s, dten_2d_s2);
  RunTestTenCtrct2DCase2(dten_2d_s, dten_2d_s2);

  auto zten_2d_s2 = zten_2d_s;
  zten_2d_s.Random(qn0);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qn0);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
  zten_2d_s.Random(qnp1);
  zten_2d_s2.Random(qnm1);
  RunTestTenCtrct2DCase1(zten_2d_s, zten_2d_s2);
  RunTestTenCtrct2DCase2(zten_2d_s, zten_2d_s2);
}


template <typename QNT, typename TenElemType>
void RunTestTenCtrct3DCase1(
    GQTensor<QNT, TenElemType> &ta,
    GQTensor<QNT, TenElemType> &tb
) {
  auto m = ta.shape[0] * ta.shape[1];
  auto n = tb.shape[1] * tb.shape[2];
  auto k1 = ta.shape[2];
  auto k2 = tb.shape[0];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n
  );
  GQTensor<QNT, TenElemType> res;
  Contract(&ta, &tb, {{2}, {0}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    GtestExpectNear(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename QNT, typename TenElemType>
void RunTestTenCtrct3DCase2(
    GQTensor<QNT, TenElemType> &ta,
    GQTensor<QNT, TenElemType> &tb
) {
  auto m = ta.shape[0];
  auto n = tb.shape[2];
  auto k1 = ta.shape[1] * ta.shape[2];
  auto k2 = tb.shape[0] * tb.shape[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  auto dense_res = new TenElemType [m * n];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  CblasGemm(
      CblasRowMajor, CblasNoTrans, CblasNoTrans,
      m, n, k1,
      1.0,
      dense_ta, k1,
      dense_tb, n,
      0.0,
      dense_res, n);
  GQTensor<QNT, TenElemType> res;
  Contract(&ta, &tb, {{1, 2}, {0, 1}}, &res);
  idx = 0;
  for (auto &coor : res.CoorsIter()) {
    GtestExpectNear(res.Elem(coor), dense_res[idx], kEpsilon);
    idx++;
  }
  delete [] dense_ta;
  delete [] dense_tb;
  delete [] dense_res;
}


template <typename QNT, typename TenElemType>
void RunTestTenCtrct3DCase3(
    GQTensor<QNT, TenElemType> &ta,
    GQTensor<QNT, TenElemType> &tb
) {
  auto m = ta.shape[0];
  auto n = tb.shape[2];
  auto k1 = ta.shape[1] * ta.shape[2];
  auto k2 = tb.shape[0] * tb.shape[1];
  auto ta_size = m * k1;
  auto tb_size = k2 * n;
  auto dense_ta =  new TenElemType [ta_size];
  auto dense_tb =  new TenElemType [tb_size];
  long idx = 0;
  for (auto &coor : ta.CoorsIter()) {
    dense_ta[idx] = ta.Elem(coor);
    idx++;
  }
  idx = 0;
  for (auto &coor : tb.CoorsIter()) {
    dense_tb[idx] = tb.Elem(coor);
    idx++;
  }
  TenElemType res_scalar = 0.0;
  for (long i = 0; i < ta_size; ++i) {
    res_scalar += (dense_ta[i] * dense_tb[i]);
  }
  GQTensor<QNT, TenElemType> res;
  Contract(&ta, &tb, {{0, 1, 2}, {0, 1, 2}}, &res);
  GtestExpectNear(res.scalar, res_scalar, kEpsilon);
  delete [] dense_ta;
  delete [] dense_tb;
}


TEST_F(TestContraction, 3DCase) {
  auto dten_3d_s2 = dten_3d_s;
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase2(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qn0);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);
  dten_3d_s.Random(qnp1);
  dten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase3(dten_3d_s, dten_3d_s2);

  auto zten_3d_s2 = zten_3d_s;
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase1(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase2(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qn0);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qn0);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s2);
  zten_3d_s.Random(qnp1);
  zten_3d_s2.Random(qnm1);
  RunTestTenCtrct3DCase3(zten_3d_s, zten_3d_s2);
}
