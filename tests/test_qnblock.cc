// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-29 10:18
* 
* Description: GraceQ/tensor project. Unittests for QNBlock object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqten/detail/utils_inl.h"   // GenAllCoors
#include "testing_utils.h"

#include <utility>
#include <algorithm>
#include <cstdio>


using namespace gqten;


template <typename QNBlockT>
void QNBlockEq(const QNBlockT &lhs, const QNBlockT &rhs) {
  EXPECT_EQ(rhs.qnscts, lhs.qnscts);
  EXPECT_EQ(rhs.ndim, lhs.ndim);
  EXPECT_EQ(rhs.shape, lhs.shape);
  EXPECT_EQ(rhs.size, lhs.size);
  GtestArrayEq(rhs.cdata(), lhs.cdata(), lhs.size);
}


using NormalQN1 = QN<NormalQNVal>;
using QNSctT = QNSector<NormalQN1>;
using QNSctSetT  = QNSectorSet<NormalQN1>;
using DQNBlockT = QNBlock<NormalQN1, GQTEN_Double>;
using ZQNBlockT = QNBlock<NormalQN1, GQTEN_Complex>;


struct TestQNBlock : public testing::Test {
  NormalQN1 qn = NormalQN1({QNCard("qn", NormalQNVal(0))});
  QNSctT qnsct1 = QNSctT(qn, 1);
  QNSctT qnsct2 = QNSctT(qn, 2);
  QNSctT qnsct3 = QNSctT(qn, 3);

  DQNBlockT qnblock_default;
  QNSctT sz0_sct1 = QNSctT(NormalQN1({QNCard("Sz", NormalQNVal(0))}), 1);
  QNSctT sz1_sct2 = QNSctT(NormalQN1({QNCard("Sz", NormalQNVal(1))}), 2);
  DQNBlockT qnblock_sz0sct1_1d = DQNBlockT({sz0_sct1});
  DQNBlockT qnblock_sz0sct1_2d = DQNBlockT({sz0_sct1, sz0_sct1});
  DQNBlockT qnblock_sz1sct2_2d = DQNBlockT({sz1_sct2, sz1_sct2});

  DQNBlockT DQNBlockT_default;
  DQNBlockT DQNBlockT_1 = DQNBlockT({qnsct1});
  DQNBlockT DQNBlockT_3 = DQNBlockT({qnsct3});
  DQNBlockT DQNBlockT_22 = DQNBlockT({qnsct2, qnsct2});
  DQNBlockT DQNBlockT_23 = DQNBlockT({qnsct2, qnsct3});
  DQNBlockT DQNBlockT_13 = DQNBlockT({qnsct1, qnsct3});
  DQNBlockT DQNBlockT_233 = DQNBlockT({qnsct2, qnsct3, qnsct3});

  ZQNBlockT ZQNBlockT_default;
  ZQNBlockT ZQNBlockT_1 = ZQNBlockT({qnsct1});
  ZQNBlockT ZQNBlockT_3 = ZQNBlockT({qnsct3});
  ZQNBlockT ZQNBlockT_22 = ZQNBlockT({qnsct2, qnsct2});
  ZQNBlockT ZQNBlockT_23 = ZQNBlockT({qnsct2, qnsct3});
  ZQNBlockT ZQNBlockT_13 = ZQNBlockT({qnsct1, qnsct3});
  ZQNBlockT ZQNBlockT_233 = ZQNBlockT({qnsct2, qnsct3, qnsct3});
};


template <typename QNBlockT>
void RunTestQNBlockNdimCase(
    const QNBlockT &qnblk, const long ndim) {
  EXPECT_EQ(qnblk.ndim, ndim);
}


TEST_F(TestQNBlock, TestNdim) {
  RunTestQNBlockNdimCase(DQNBlockT_default, 0);
  RunTestQNBlockNdimCase(DQNBlockT_1, 1);
  RunTestQNBlockNdimCase(DQNBlockT_3, 1);
  RunTestQNBlockNdimCase(DQNBlockT_22, 2);
  RunTestQNBlockNdimCase(DQNBlockT_23, 2);
  RunTestQNBlockNdimCase(DQNBlockT_13, 2);
  RunTestQNBlockNdimCase(DQNBlockT_233, 3);

  RunTestQNBlockNdimCase(ZQNBlockT_default, 0);
  RunTestQNBlockNdimCase(ZQNBlockT_1, 1);
  RunTestQNBlockNdimCase(ZQNBlockT_3, 1);
  RunTestQNBlockNdimCase(ZQNBlockT_22, 2);
  RunTestQNBlockNdimCase(ZQNBlockT_23, 2);
  RunTestQNBlockNdimCase(ZQNBlockT_13, 2);
  RunTestQNBlockNdimCase(ZQNBlockT_233, 3);
}


template <typename QNBlockT>
void RunTestQNBlockShapeCase(
    const QNBlockT &qnblk, const std::vector<long> &shape) {
  EXPECT_EQ(qnblk.shape, shape);
}


TEST_F(TestQNBlock, TestShape) {
  RunTestQNBlockShapeCase(DQNBlockT_default, {});
  RunTestQNBlockShapeCase(DQNBlockT_1, {1});
  RunTestQNBlockShapeCase(DQNBlockT_3, {3});
  RunTestQNBlockShapeCase(DQNBlockT_22, {2, 2});
  RunTestQNBlockShapeCase(DQNBlockT_23, {2, 3});
  RunTestQNBlockShapeCase(DQNBlockT_13, {1, 3});
  RunTestQNBlockShapeCase(DQNBlockT_233, {2, 3 ,3});

  RunTestQNBlockShapeCase(ZQNBlockT_default, {});
  RunTestQNBlockShapeCase(ZQNBlockT_1, {1});
  RunTestQNBlockShapeCase(ZQNBlockT_3, {3});
  RunTestQNBlockShapeCase(ZQNBlockT_22, {2, 2});
  RunTestQNBlockShapeCase(ZQNBlockT_23, {2, 3});
  RunTestQNBlockShapeCase(ZQNBlockT_13, {1, 3});
  RunTestQNBlockShapeCase(ZQNBlockT_233, {2, 3 ,3});
}


template <typename QNT, typename ElemType>
void RunTestQNBlockElemAssignmentCase(
    const QNBlock<QNT, ElemType> &qnblk_init,
    const std::vector<ElemType> elems,
    const std::vector<std::vector<long>> coors) {
  auto qnblk = qnblk_init;
  for (size_t i = 0; i < elems.size(); ++i) {
    qnblk(coors[i]) = elems[i];
  }
  for (auto coor : GenAllCoors(qnblk.shape)) {
    auto coor_it = std::find(coors.cbegin(), coors.cend(), coor); 
    if (coor_it != coors.end()) {
      auto elem_idx = std::distance(coors.begin(), coor_it);
      EXPECT_EQ(qnblk(coor), elems[elem_idx]);
    } else {
      EXPECT_EQ(qnblk(coor), ElemType(0.0));
    }
  }
}


TEST_F(TestQNBlock, TestElemAssignment) {
  RunTestQNBlockElemAssignmentCase(DQNBlockT_1, {1.0}, {{0}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_3, {1.0}, {{0}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_3, {1.0, 2.0}, {{0}, {1}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_3, {1.0, 2.0}, {{1}, {2}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_13, {1.0}, {{0, 1}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_22, {1.0}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(DQNBlockT_233, {1.0}, {{0, 1, 2}});
  RunTestQNBlockElemAssignmentCase(
      DQNBlockT_233,
      {1.0, 2.0}, {{1, 0, 2}, {0, 2, 1}});

  RunTestQNBlockElemAssignmentCase(ZQNBlockT_1, {GQTEN_Complex(0.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(ZQNBlockT_1, {GQTEN_Complex(1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_1,
      {GQTEN_Complex(1.0, 0.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_1,
      {GQTEN_Complex(0.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_1,
      {GQTEN_Complex(1.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_3,
      {GQTEN_Complex(1.0, 1.0)}, {{0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_3,
      {GQTEN_Complex(1.0, 1.0)}, {{1}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_3,
      {GQTEN_Complex(1.0, 1.0)}, {{2}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_3,
      {GQTEN_Complex(1.0, 0.1), GQTEN_Complex(2.0, 0.2)}, {{0}, {1}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 1}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_13,
      {GQTEN_Complex(1.0, 1.0)}, {{0, 2}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_13,
      {GQTEN_Complex(1.0, 0.1), GQTEN_Complex(2.0, 0.2)}, {{0, 0}, {0, 1}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_22,
      {GQTEN_Complex(1.0, 0.1)}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_22, {GQTEN_Complex(1.0, 0.1)}, {{1, 0}});
  RunTestQNBlockElemAssignmentCase(
      ZQNBlockT_233, {GQTEN_Complex(1.0, 0.1)}, {{0, 1, 2}});
}


TEST_F(TestQNBlock, TestPartialHash) {
  EXPECT_EQ(
      qnblock_sz0sct1_1d.PartHash({0}),
      QNSctSetT({qnblock_sz0sct1_1d.qnscts[0]}).Hash()
  );
  EXPECT_EQ(
      qnblock_sz0sct1_2d.PartHash({0, 1}),
      QNSctSetT(
          {qnblock_sz0sct1_2d.qnscts[0], qnblock_sz0sct1_2d.qnscts[1]}
      ).Hash()
  );
}


TEST_F(TestQNBlock, TestQNSctTSetHash) {
  EXPECT_EQ(qnblock_default.QNSectorSetHash(), 0);
  EXPECT_EQ(
      qnblock_sz0sct1_1d.QNSectorSetHash(),
      QNSctSetT(qnblock_sz0sct1_1d.qnscts).Hash()
  );
  EXPECT_EQ(
      qnblock_sz1sct2_2d.QNSectorSetHash(),
      QNSctSetT(qnblock_sz1sct2_2d.qnscts).Hash()
  );
}


template <typename QNBlockT>
void RunTestQNBlockHashMethodsCase(
    const QNBlockT &qnblk,
    const std::vector<long> &part_axes
) {
  EXPECT_EQ(qnblk.QNSectorSetHash(), QNSctSetT(qnblk.qnscts).Hash());

  std::vector<QNSctT> part_qnscts;
  for (auto axis : part_axes) {
    part_qnscts.push_back(qnblk.qnscts[axis]);
  }
  EXPECT_EQ(qnblk.PartHash(part_axes), QNSctSetT(part_qnscts).Hash());
}


TEST_F(TestQNBlock, TestHashMethods) {
  /* Why it fail? Because VecHasher({}) !=0
   * TODO: Let VecHasher({}) == 0
   */
  //RunTestQNBlockHashMethodsCase(DQNBlockT_default, {});
  RunTestQNBlockHashMethodsCase(DQNBlockT_3, {0});
  RunTestQNBlockHashMethodsCase(DQNBlockT_23, {0});
  RunTestQNBlockHashMethodsCase(DQNBlockT_23, {1});
  RunTestQNBlockHashMethodsCase(DQNBlockT_23, {0, 1});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {0});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {1});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {2});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {0, 1});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {1, 2});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {0, 2});
  RunTestQNBlockHashMethodsCase(DQNBlockT_233, {0, 1, 2});

  RunTestQNBlockHashMethodsCase(ZQNBlockT_3, {0});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_23, {0});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_23, {1});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_23, {0, 1});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {0});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {1});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {2});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {0, 1});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {1, 2});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {0, 2});
  RunTestQNBlockHashMethodsCase(ZQNBlockT_233, {0, 1, 2});
}


// Test rand a QNBlock.
template <typename QNT, typename ElemType>
void RunTestRanDQNBlockTCase(QNBlock<QNT, ElemType> &qnblk) {
  auto size = qnblk.size;
  auto rand_array = new ElemType[size]();
  srand(0);
  for (long i = 0; i < size; ++i) {
    Rand(rand_array[i]);
  }
  srand(0);
  qnblk.Random();
  GtestArrayEq(rand_array, qnblk.cdata(), size);
  delete [] rand_array;
}


TEST_F(TestQNBlock, TestRanDQNBlockT) {
  RunTestRanDQNBlockTCase(DQNBlockT_default);
  RunTestRanDQNBlockTCase(DQNBlockT_1);
  RunTestRanDQNBlockTCase(DQNBlockT_3);
  RunTestRanDQNBlockTCase(DQNBlockT_13);
  RunTestRanDQNBlockTCase(DQNBlockT_22);
  RunTestRanDQNBlockTCase(DQNBlockT_23);
  RunTestRanDQNBlockTCase(DQNBlockT_233);

  RunTestRanDQNBlockTCase(ZQNBlockT_default);
  RunTestRanDQNBlockTCase(ZQNBlockT_1);
  RunTestRanDQNBlockTCase(ZQNBlockT_3);
  RunTestRanDQNBlockTCase(ZQNBlockT_13);
  RunTestRanDQNBlockTCase(ZQNBlockT_22);
  RunTestRanDQNBlockTCase(ZQNBlockT_23);
  RunTestRanDQNBlockTCase(ZQNBlockT_233);
}


// Test QNBlock transpose.
template <typename QNBlkT>
void RunTestQNBlockTransposeCase(
    const QNBlkT &blk_init, const std::vector<long> &axes) {
  auto blk = blk_init;
  blk.Random();
  auto transed_blk = blk;
  transed_blk.Transpose(axes);
  for (size_t i = 0; i < axes.size(); ++i) {
    EXPECT_EQ(transed_blk.shape[i], blk.shape[axes[i]]);
  }
  for (auto blk_coors : GenAllCoors(blk.shape)) {
    EXPECT_EQ(transed_blk(TransCoors(blk_coors, axes)), blk(blk_coors));
  }
}


TEST_F(TestQNBlock, TestQNBlockTranspose) {
  RunTestQNBlockTransposeCase(DQNBlockT_1, {0});
  RunTestQNBlockTransposeCase(DQNBlockT_3, {0});
  RunTestQNBlockTransposeCase(DQNBlockT_13, {0, 1});
  RunTestQNBlockTransposeCase(DQNBlockT_13, {1, 0});
  RunTestQNBlockTransposeCase(DQNBlockT_22, {0, 1});
  RunTestQNBlockTransposeCase(DQNBlockT_22, {1, 0});
  RunTestQNBlockTransposeCase(DQNBlockT_23, {0, 1});
  RunTestQNBlockTransposeCase(DQNBlockT_23, {1, 0});
  RunTestQNBlockTransposeCase(DQNBlockT_233, {0, 1, 2});
  RunTestQNBlockTransposeCase(DQNBlockT_233, {1, 0, 2});
  RunTestQNBlockTransposeCase(DQNBlockT_233, {0, 2, 1});
  RunTestQNBlockTransposeCase(DQNBlockT_233, {2, 0, 1});

  RunTestQNBlockTransposeCase(ZQNBlockT_1, {0});
  RunTestQNBlockTransposeCase(ZQNBlockT_3, {0});
  RunTestQNBlockTransposeCase(ZQNBlockT_13, {0, 1});
  RunTestQNBlockTransposeCase(ZQNBlockT_13, {1, 0});
  RunTestQNBlockTransposeCase(ZQNBlockT_22, {0, 1});
  RunTestQNBlockTransposeCase(ZQNBlockT_22, {1, 0});
  RunTestQNBlockTransposeCase(ZQNBlockT_23, {0, 1});
  RunTestQNBlockTransposeCase(ZQNBlockT_23, {1, 0});
  RunTestQNBlockTransposeCase(ZQNBlockT_233, {0, 1, 2});
  RunTestQNBlockTransposeCase(ZQNBlockT_233, {1, 0, 2});
  RunTestQNBlockTransposeCase(ZQNBlockT_233, {0, 2, 1});
  RunTestQNBlockTransposeCase(ZQNBlockT_233, {2, 0, 1});
}


template <typename QNBlockT>
void RunTestQNBlockCopyAndMoveConstructorsCase(QNBlockT &qnblk) {
  qnblk.Random();

  QNBlockT qnblk_copyed(qnblk);
  QNBlockEq(qnblk_copyed, qnblk);
  auto qnblk_copyed2 = qnblk;
  QNBlockEq(qnblk_copyed2, qnblk);

  auto qnblk_tomove = qnblk;
  QNBlockT qnblk_moved(std::move(qnblk_tomove));
  QNBlockEq(qnblk_moved, qnblk);
  EXPECT_EQ(qnblk_tomove.cdata(), nullptr);
  auto qnblk_tomove2 = qnblk;
  auto qnblk_moved2 = std::move(qnblk_tomove2);
  QNBlockEq(qnblk_moved2, qnblk);
  EXPECT_EQ(qnblk_tomove2.cdata(), nullptr);
}


TEST_F(TestQNBlock, TestCopyAndMoveConstructors) {
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_default);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_1);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_3);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_13);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_22);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_23);
  RunTestQNBlockCopyAndMoveConstructorsCase(DQNBlockT_233);

  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_default);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_1);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_3);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_13);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_22);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_23);
  RunTestQNBlockCopyAndMoveConstructorsCase(ZQNBlockT_233);
}


template <typename QNBlockT>
void RunTestQNBlockFileIOCase(QNBlockT &qnblk) {
  qnblk.Random();

  std::string file = "test.qnblk";
  std::ofstream out(file, std::ofstream::binary);
  out << qnblk;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNBlockT qnblk_cpy;
  in >> qnblk_cpy;
  in.close();

  EXPECT_EQ(qnblk_cpy.ndim, qnblk.ndim);
  EXPECT_EQ(qnblk_cpy.size, qnblk.size);
  EXPECT_EQ(qnblk_cpy.shape, qnblk.shape);
  EXPECT_EQ(qnblk_cpy.qnscts, qnblk.qnscts);
  GtestArrayEq(qnblk_cpy.cdata(), qnblk.cdata(), qnblk_cpy.size);
}


TEST_F(TestQNBlock, FileIO) {
  RunTestQNBlockFileIOCase(DQNBlockT_default);
  RunTestQNBlockFileIOCase(DQNBlockT_1);
  RunTestQNBlockFileIOCase(DQNBlockT_3);
  RunTestQNBlockFileIOCase(DQNBlockT_13);
  RunTestQNBlockFileIOCase(DQNBlockT_22);
  RunTestQNBlockFileIOCase(DQNBlockT_23);
  RunTestQNBlockFileIOCase(DQNBlockT_233);

  RunTestQNBlockFileIOCase(ZQNBlockT_default);
  RunTestQNBlockFileIOCase(ZQNBlockT_1);
  RunTestQNBlockFileIOCase(ZQNBlockT_3);
  RunTestQNBlockFileIOCase(ZQNBlockT_13);
  RunTestQNBlockFileIOCase(ZQNBlockT_22);
  RunTestQNBlockFileIOCase(ZQNBlockT_23);
  RunTestQNBlockFileIOCase(ZQNBlockT_233);
}
