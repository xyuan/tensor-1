// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 13:39
* 
* Description: GraceQ/tensor project. Unittests for GQTensor object.
*/
#include "gqten/gqten.h"
#include "gqten/detail/utils_inl.h"   // GenAllCoors
#include "testing_utils.h"
#include "gtest/gtest.h"

#include <utility>
#include <string>
#include <type_traits>                // remove_reference
#include <cmath>
#include <cstdio>


using namespace gqten;


using NormalQN1 = QN<NormalQNVal>;
using QNSctT = QNSector<NormalQN1>;
using IndexT = Index<NormalQN1>;
using DGQTenT = GQTensor<NormalQN1, GQTEN_Double>;
using ZGQTenT = GQTensor<NormalQN1, GQTEN_Complex>;


struct TestGQTensor : public testing::Test {
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

  DGQTenT dten_default = DGQTenT();
  DGQTenT dten_1d_s = DGQTenT({idx_out_s});
  DGQTenT dten_1d_l = DGQTenT({idx_out_l});
  DGQTenT dten_2d_s = DGQTenT({idx_in_s, idx_out_s});
  DGQTenT dten_2d_l = DGQTenT({idx_in_l, idx_out_l});
  DGQTenT dten_3d_s = DGQTenT({idx_in_s, idx_out_s, idx_out_s});
  DGQTenT dten_3d_l = DGQTenT({idx_in_l, idx_out_l, idx_out_l});
  ZGQTenT zten_default = ZGQTenT();
  ZGQTenT zten_1d_s = ZGQTenT({idx_out_s});
  ZGQTenT zten_1d_l = ZGQTenT({idx_out_l});
  ZGQTenT zten_2d_s = ZGQTenT({idx_in_s, idx_out_s});
  ZGQTenT zten_2d_l = ZGQTenT({idx_in_l, idx_out_l});
  ZGQTenT zten_3d_s = ZGQTenT({idx_in_s, idx_out_s, idx_out_s});
  ZGQTenT zten_3d_l = ZGQTenT({idx_in_l, idx_out_l, idx_out_l});
};


template <typename GQTensorT, typename IdxT>
void RunTestGQTensorCommonConstructorCase(
    const GQTensorT &t,
    const std::vector<IdxT> &indexes) {
  EXPECT_EQ(t.indexes, indexes);
}


TEST_F(TestGQTensor, TestCommonConstructor) {
  RunTestGQTensorCommonConstructorCase<DGQTenT, IndexT>(
      dten_default, {}
  );
  RunTestGQTensorCommonConstructorCase<DGQTenT, IndexT>(
      dten_1d_s, {idx_out_s}
  );
  RunTestGQTensorCommonConstructorCase<DGQTenT, IndexT>(
      dten_2d_s, {idx_in_s, idx_out_s}
  );
  RunTestGQTensorCommonConstructorCase<DGQTenT, IndexT>(
      dten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s}
  );

  RunTestGQTensorCommonConstructorCase<ZGQTenT, IndexT>(
      zten_default, {}
  );
  RunTestGQTensorCommonConstructorCase<ZGQTenT, IndexT>(
      zten_1d_s, {idx_out_s}
  );
  RunTestGQTensorCommonConstructorCase<ZGQTenT, IndexT>(
      zten_2d_s, {idx_in_s, idx_out_s}
  );
  RunTestGQTensorCommonConstructorCase<ZGQTenT, IndexT>(
      zten_3d_s,
      {idx_in_s, idx_out_s, idx_out_s}
  );
}


template <typename QNT, typename ElemType>
void RunTestGQTensorElemAssignmentCase(
    const GQTensor<QNT, ElemType> &t_init,
    const std::vector<ElemType> elems,
    const std::vector<std::vector<long>> coors) {
  auto t = t_init;
  for (size_t i = 0; i < elems.size(); ++i) {
    t(coors[i]) = elems[i];
  }
  for (auto coor : GenAllCoors(t.shape)) {
    auto coor_it = std::find(coors.cbegin(), coors.cend(), coor);
    if (coor_it != coors.end()) {
      auto elem_idx = std::distance(coors.cbegin(), coor_it);
      EXPECT_EQ(t.Elem(coor), elems[elem_idx]);
    } else {
      EXPECT_EQ(t.Elem(coor), ElemType(0.0));
    }
  }
}


TEST_F(TestGQTensor, TestElemAssignment) {
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand()}, {{0}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand()}, {{1}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{0}, {1}});
  RunTestGQTensorElemAssignmentCase(dten_1d_s, {drand(), drand()}, {{1}, {2}});
  RunTestGQTensorElemAssignmentCase(dten_2d_s, {drand()}, {{0, 0}});
  RunTestGQTensorElemAssignmentCase(dten_2d_s, {drand()}, {{2, 3}});
  RunTestGQTensorElemAssignmentCase(
      dten_2d_s,
      {drand(), drand()},
      {{2, 3}, {1, 7}});
  RunTestGQTensorElemAssignmentCase(dten_3d_s, {drand()}, {{0, 0, 0}});
  RunTestGQTensorElemAssignmentCase(dten_3d_s, {drand()}, {{2, 3, 4}});
  RunTestGQTensorElemAssignmentCase(
      dten_3d_s,
      {drand(), drand()},
      {{2, 3, 5}, {1, 7, 4}}
  );

  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{0}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand()},
      {{1}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{0}, {1}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_1d_s,
      {zrand(), zrand()},
      {{1}, {2}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{0, 0}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand()},
      {{2, 3}}
  );
  RunTestGQTensorElemAssignmentCase(
      zten_2d_s,
      {zrand(), zrand()},
      {{2, 3}, {1, 7}}
  );
  RunTestGQTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{0, 0, 0}});
  RunTestGQTensorElemAssignmentCase(zten_3d_s, {zrand()}, {{2, 3, 4}});
  RunTestGQTensorElemAssignmentCase(
      zten_3d_s,
      {zrand(), zrand()},
      {{2, 3, 5}, {1, 7, 4}}
  );
}


template <typename QNT, typename ElemType>
void RunTestGQTensorRandomCase(
    GQTensor<QNT, ElemType> &t,
    const QNT &div,
    const std::vector<QNSectorVec<QNT>> &qnscts_set) {
  std::vector<QNSectorVec<QNT>> had_qnscts_set;
  srand(0);
  t.Random(div);
  srand(0);
  EXPECT_EQ(t.cblocks().size(), qnscts_set.size());
  for (auto &blk : t.cblocks()) {
    auto qnscts_it = std::find(
                         qnscts_set.begin(), qnscts_set.end(), blk->qnscts);
    auto had_qnscts_it = std::find(
                             had_qnscts_set.begin(), had_qnscts_set.end(),
                             blk->qnscts);
    EXPECT_TRUE(
        (qnscts_it != qnscts_set.end()) &&
        (had_qnscts_it == had_qnscts_set.end()));
    had_qnscts_set.push_back(*qnscts_it);
    for (size_t i = 0; i < blk->size; ++i) {
      EXPECT_EQ(blk->cdata()[i], RandT<ElemType>());
    }
  }
}


TEST_F(TestGQTensor, Random) {
  RunTestGQTensorRandomCase(dten_1d_s, qn0, {{qnsct0_s}});
  RunTestGQTensorRandomCase(dten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestGQTensorRandomCase(dten_1d_l, qn0, {{qnsct0_l}});
  RunTestGQTensorRandomCase(dten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s},
        {qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnm1,
      {
        {qnsct0_s, qnsctm1_s},
        {qnsctp1_s, qnsct0_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l},
        {qnsctp1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnm1,
      {
        {qnsct0_l, qnsctm1_l},
        {qnsctp1_l, qnsct0_l}
      }
  );
  RunTestGQTensorRandomCase(
      dten_2d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctm1_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsct0_s},
        {qnsctp1_s, qnsct0_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s, qnsct0_s},
        {qnsctm1_s, qnsctm1_s, qnsctp1_s},
        {qnsctm1_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctp1_s, qnsct0_s},
        {qnsct0_s, qnsct0_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctp1_s},
        {qnsct0_s, qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctm1_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsct0_l},
        {qnsctp1_l, qnsct0_l, qnsctp1_l} 
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l, qnsct0_l},
        {qnsctm1_l, qnsctm1_l, qnsctp1_l},
        {qnsctm1_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctp1_l, qnsct0_l},
        {qnsct0_l, qnsct0_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      dten_3d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctp1_l},
        {qnsct0_l, qnsctp1_l, qnsctp1_l}
      }
  );

  RunTestGQTensorRandomCase(zten_1d_s, qn0, {{qnsct0_s}});
  RunTestGQTensorRandomCase(zten_1d_s, qnp1, {{qnsctp1_s}});
  RunTestGQTensorRandomCase(zten_1d_l, qn0, {{qnsct0_l}});
  RunTestGQTensorRandomCase(zten_1d_l, qnp1, {{qnsctp1_l}});
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s},
        {qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnm1,
      {
        {qnsct0_s, qnsctm1_s},
        {qnsctp1_s, qnsct0_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l},
        {qnsctp1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnm1,
      {
        {qnsct0_l, qnsctm1_l},
        {qnsctp1_l, qnsct0_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_2d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qn0,
      {
        {qnsctm1_s, qnsctm1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctm1_s},
        {qnsct0_s, qnsct0_s, qnsct0_s},
        {qnsct0_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctm1_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsct0_s},
        {qnsctp1_s, qnsct0_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qnp1,
      {
        {qnsctm1_s, qnsct0_s, qnsct0_s},
        {qnsctm1_s, qnsctm1_s, qnsctp1_s},
        {qnsctm1_s, qnsctp1_s, qnsctm1_s},
        {qnsct0_s, qnsctp1_s, qnsct0_s},
        {qnsct0_s, qnsct0_s, qnsctp1_s},
        {qnsctp1_s, qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_s,
      qnp2,
      {
        {qnsctm1_s, qnsctp1_s, qnsct0_s},
        {qnsctm1_s, qnsct0_s, qnsctp1_s},
        {qnsct0_s, qnsctp1_s, qnsctp1_s}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qn0,
      {
        {qnsctm1_l, qnsctm1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctm1_l},
        {qnsct0_l, qnsct0_l, qnsct0_l},
        {qnsct0_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctm1_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsct0_l},
        {qnsctp1_l, qnsct0_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qnp1,
      {
        {qnsctm1_l, qnsct0_l, qnsct0_l},
        {qnsctm1_l, qnsctm1_l, qnsctp1_l},
        {qnsctm1_l, qnsctp1_l, qnsctm1_l},
        {qnsct0_l, qnsctp1_l, qnsct0_l},
        {qnsct0_l, qnsct0_l, qnsctp1_l},
        {qnsctp1_l, qnsctp1_l, qnsctp1_l}
      }
  );
  RunTestGQTensorRandomCase(
      zten_3d_l,
      qnp2,
      {
        {qnsctm1_l, qnsctp1_l, qnsct0_l},
        {qnsctm1_l, qnsct0_l, qnsctp1_l},
        {qnsct0_l, qnsctp1_l, qnsctp1_l}
      }
  );
}


template <typename GQTensorT>
void RunTestGQTensorEqCase(
    const GQTensorT &lhs, const GQTensorT &rhs, int test_eq_flag = 1) {
  if (test_eq_flag) {
    EXPECT_TRUE(lhs == rhs);
  } else {
    EXPECT_TRUE(lhs != rhs);
  }
}


TEST_F(TestGQTensor, TestEq) {
  RunTestGQTensorEqCase(dten_default, dten_default);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s);
  dten_1d_s.Random(qn0);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s);
  decltype(dten_1d_s) dten_1d_s2(dten_1d_s.indexes);
  dten_1d_s2.Random(qnp1);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_s2, 0);
  RunTestGQTensorEqCase(dten_1d_s, dten_1d_l, 0);
  RunTestGQTensorEqCase(dten_1d_s, dten_2d_s, 0);

  RunTestGQTensorEqCase(zten_default, zten_default);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s);
  zten_1d_s.Random(qn0);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s);
  decltype(zten_1d_s) zten_1d_s2(zten_1d_s.indexes);
  zten_1d_s2.Random(qnp1);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_s2, 0);
  RunTestGQTensorEqCase(zten_1d_s, zten_1d_l, 0);
  RunTestGQTensorEqCase(zten_1d_s, zten_2d_s, 0);
}


template <typename GQTensorT>
void RunTestGQTensorCopyAndMoveConstructorsCase(const GQTensorT &t) {
  GQTensorT gqten_cpy(t);
  EXPECT_EQ(gqten_cpy, t);
  auto gqten_cpy2 = t;
  EXPECT_EQ(gqten_cpy2, t);

  GQTensorT gqten_tomove(t);
  GQTensorT gqten_moved(std::move(gqten_tomove));
  EXPECT_EQ(gqten_moved, t);
  EXPECT_EQ(
      gqten_tomove.cblocks(),
      std::vector<
          typename std::remove_reference<
              decltype(t.cblocks())>::type::value_type>{});
  GQTensorT gqten_tomove2(t);
  auto gqten_moved2 = std::move(gqten_tomove2);
  EXPECT_EQ(gqten_moved2, t);
  EXPECT_EQ(
      gqten_tomove2.cblocks(),
      std::vector<
          typename std::remove_reference<
              decltype(t.cblocks())>::type::value_type>{});
}


TEST_F(TestGQTensor, TestCopyAndMoveConstructors) {
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(dten_3d_s);

  RunTestGQTensorCopyAndMoveConstructorsCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorCopyAndMoveConstructorsCase(zten_3d_s);
}


template <typename GQTensorT>
void RunTestGQTensorTransposeCase(
    const GQTensorT &t, const std::vector<long> &axes) {
  auto transed_t = t;
  transed_t.Transpose(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    EXPECT_EQ(transed_t.indexes[i], t.indexes[axes[i]]);
    EXPECT_EQ(transed_t.shape[i], t.shape[axes[i]]);
  }
  for (auto &coors : t.CoorsIter()) {
    EXPECT_EQ(transed_t.Elem(TransCoors(coors, axes)), t.Elem(coors));
  }
}


TEST_F(TestGQTensor, TestTranspose) {
  RunTestGQTensorTransposeCase(dten_default, {});
  dten_1d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_1d_s, {0});
  dten_2d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(dten_2d_s, {1, 0});
  dten_2d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(dten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(dten_2d_s, {1, 0});
  dten_3d_s.Random(qn0);
  RunTestGQTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {2, 0, 1});
  dten_3d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(dten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(dten_3d_s, {2, 0, 1});

  RunTestGQTensorTransposeCase(zten_default, {});
  zten_1d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_1d_s, {0});
  zten_2d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(zten_2d_s, {1, 0});
  zten_2d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(zten_2d_s, {0, 1});
  RunTestGQTensorTransposeCase(zten_2d_s, {1, 0});
  zten_3d_s.Random(qn0);
  RunTestGQTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {2, 0, 1});
  zten_3d_s.Random(qnp1);
  RunTestGQTensorTransposeCase(zten_3d_s, {0, 1, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {1, 0, 2});
  RunTestGQTensorTransposeCase(zten_3d_s, {2, 0, 1});
}


template <typename GQTensorT>
void RunTestGQTensorNormalizeCase(GQTensorT &t) {
  t.Normalize();
  auto norm = 0.0;
  for (auto &blk : t.cblocks()) {
    for (long i = 0; i < blk->size; ++i) {
      norm += std::norm(blk->cdata()[i]);
    }
  }
  EXPECT_NEAR(norm, 1.0, kEpsilon);
}


TEST_F(TestGQTensor, TestNormalize) {
  dten_1d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestGQTensorNormalizeCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(dten_3d_s);

  zten_1d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestGQTensorNormalizeCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorNormalizeCase(zten_3d_s);
}


template <typename GQTensorT>
void RunTestGQTensorDagCase(const GQTensorT &t) {
  auto t_dag = Dag(t);
  for (size_t i = 0; i < t.indexes.size(); ++i) {
    EXPECT_EQ(t_dag.indexes[i], InverseIndex(t.indexes[i]));
  }
  for (auto &coor : GenAllCoors(t.shape)) {
    EXPECT_EQ(t_dag.Elem(coor), std::conj(t.Elem(coor)));
  }
}


TEST_F(TestGQTensor, TestDag) {
  RunTestGQTensorDagCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorDagCase(dten_3d_s);

  RunTestGQTensorDagCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorDagCase(zten_3d_s);
}


template <typename GQTensorT, typename QNT>
void RunTestGQTensorDivCase(GQTensorT &t, const QNT &div) {
  t.Random(div);
  EXPECT_EQ(Div(t), div);
}


TEST_F(TestGQTensor, TestDiv) {
  RunTestGQTensorDivCase(dten_1d_s, qn0);
  RunTestGQTensorDivCase(dten_2d_s, qn0);
  RunTestGQTensorDivCase(dten_2d_s, qnp1);
  RunTestGQTensorDivCase(dten_2d_s, qnm1);
  RunTestGQTensorDivCase(dten_2d_s, qnp2);
  RunTestGQTensorDivCase(dten_3d_s, qn0);
  RunTestGQTensorDivCase(dten_3d_s, qnp1);
  RunTestGQTensorDivCase(dten_3d_s, qnm1);
  RunTestGQTensorDivCase(dten_3d_s, qnp2);

  RunTestGQTensorDivCase(zten_1d_s, qn0);
  RunTestGQTensorDivCase(zten_2d_s, qn0);
  RunTestGQTensorDivCase(zten_2d_s, qnp1);
  RunTestGQTensorDivCase(zten_2d_s, qnm1);
  RunTestGQTensorDivCase(zten_2d_s, qnp2);
  RunTestGQTensorDivCase(zten_3d_s, qn0);
  RunTestGQTensorDivCase(zten_3d_s, qnp1);
  RunTestGQTensorDivCase(zten_3d_s, qnm1);
  RunTestGQTensorDivCase(zten_3d_s, qnp2);
}


template <typename GQTensorT>
void RunTestGQTensorSumCase(GQTensorT &lhs, GQTensorT &rhs) {
  auto sum1 = lhs + rhs;
  auto sum2 = lhs;
  sum2 += rhs;
  EXPECT_EQ(sum1, sum2);
}


TEST_F(TestGQTensor, TestSummation) {
  dten_1d_s.Random(qn0);
  RunTestGQTensorSumCase(dten_1d_s, dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorSumCase(dten_2d_s, dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorSumCase(dten_3d_s, dten_3d_s);

  zten_1d_s.Random(qn0);
  RunTestGQTensorSumCase(zten_1d_s, zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorSumCase(zten_2d_s, zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorSumCase(zten_3d_s, zten_3d_s);
}


template <typename QNT, typename ElemType>
void RunTestGQTensorDotMultiCase(
    const GQTensor<QNT, ElemType> &t, const ElemType scalar) {
  auto multied_t = scalar * t;
  for (auto &coor : GenAllCoors(t.shape)) {
    EXPECT_EQ(multied_t.Elem(coor), scalar * t.Elem(coor));
  }
}


TEST_F(TestGQTensor, TestDotMultiplication) {
  dten_default.scalar = 1.0;
  auto rand_d = drand();
  auto multied_ten = rand_d * dten_default;
  EXPECT_DOUBLE_EQ(multied_ten.scalar, rand_d);

  dten_1d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(dten_1d_s, drand());
  dten_2d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(dten_2d_s, drand());
  dten_3d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(dten_3d_s, drand());

  zten_default.scalar = GQTEN_Complex(1.0);
  auto rand_z = zrand();
  auto multied_zten = rand_z * zten_default;
  EXPECT_EQ(multied_zten.scalar, rand_z);

  zten_1d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(zten_1d_s, zrand());
  zten_2d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(zten_2d_s, zrand());
  zten_3d_s.Random(qn0);
  RunTestGQTensorDotMultiCase(zten_3d_s, zrand());
}


template <typename QNT, typename TenElemType>
void RunTestGQTensorToComplexCase(
    GQTensor<QNT, TenElemType> &t,
    const QNT & div,
    unsigned int rand_seed) {
  srand(rand_seed);
  t.Random(div);
  auto zten = ToComplex(t);
  for (auto &coor : t.CoorsIter()) {
    EXPECT_DOUBLE_EQ(zten.Elem(coor).real(), t.Elem(coor));
  }
}


TEST_F(TestGQTensor, TestToComplex) {
  srand(0);
  dten_default.scalar = drand();
  auto zten = ToComplex(dten_default);
  srand(0);
  EXPECT_DOUBLE_EQ(zten.scalar.real(), drand());

  RunTestGQTensorToComplexCase(dten_1d_s, qn0, 0);
  RunTestGQTensorToComplexCase(dten_1d_s, qn0, 1);
  RunTestGQTensorToComplexCase(dten_1d_s, qnp1, 0);
  RunTestGQTensorToComplexCase(dten_1d_s, qnp1, 1);
  RunTestGQTensorToComplexCase(dten_2d_s, qn0, 0);
  RunTestGQTensorToComplexCase(dten_2d_s, qn0, 1);
  RunTestGQTensorToComplexCase(dten_2d_s, qnp1, 0);
  RunTestGQTensorToComplexCase(dten_2d_s, qnp1, 1);
  RunTestGQTensorToComplexCase(dten_3d_s, qn0, 0);
  RunTestGQTensorToComplexCase(dten_3d_s, qn0, 1);
  RunTestGQTensorToComplexCase(dten_3d_s, qnp1, 0);
  RunTestGQTensorToComplexCase(dten_3d_s, qnp1, 1);
}


template <typename GQTensorT>
void RunTestGQTensorFileIOCase(const GQTensorT &t) {
  std::string file = "test.gqten";
  std::ofstream out(file, std::ofstream::binary);
  out << t;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  GQTensorT t_cpy;
  in >> t_cpy;
  in.close();
  EXPECT_EQ(t_cpy, t);
}


TEST_F(TestGQTensor, FileIO) {
  RunTestGQTensorFileIOCase(dten_default);
  dten_1d_s.Random(qn0);
  RunTestGQTensorFileIOCase(dten_1d_s);
  dten_1d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(dten_1d_s);
  dten_2d_s.Random(qn0);
  RunTestGQTensorFileIOCase(dten_2d_s);
  dten_2d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(dten_2d_s);
  dten_3d_s.Random(qn0);
  RunTestGQTensorFileIOCase(dten_3d_s);
  dten_3d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(dten_3d_s);

  RunTestGQTensorFileIOCase(zten_default);
  zten_1d_s.Random(qn0);
  RunTestGQTensorFileIOCase(zten_1d_s);
  zten_1d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(zten_1d_s);
  zten_2d_s.Random(qn0);
  RunTestGQTensorFileIOCase(zten_2d_s);
  zten_2d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(zten_2d_s);
  zten_3d_s.Random(qn0);
  RunTestGQTensorFileIOCase(zten_3d_s);
  zten_3d_s.Random(qnp1);
  RunTestGQTensorFileIOCase(zten_3d_s);
}
