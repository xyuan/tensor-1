// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-25 22:44
* 
* Description: GraceQ/tensor project. Unit tests for QNSector class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <assert.h>

#ifdef Release
  #define NDEBUG
#endif


using namespace gqten;


using NormalQN1 = QN<NormalQNVal>;
using NormalQN2 = QN<NormalQNVal, NormalQNVal>;

using QNSctT1 = QNSector<NormalQN1>;
using QNSctT2 = QNSector<NormalQN2>;

struct TestQNSector : public testing::Test {
  QNSctT1 qnsct1_default = QNSctT1();
  QNSctT1 qnsct1 = QNSctT1(NormalQN1({QNCard("Sz", NormalQNVal(1))}), 1);
  QNSctT2 qnsct2 = QNSctT2(
                       NormalQN2({
                           QNCard("Sz", NormalQNVal(-1)),
                           QNCard("N",  NormalQNVal(1))}),
                       2);
};


TEST_F(TestQNSector, DataMembers) {
  EXPECT_EQ(qnsct1_default.qn, NormalQN1());
  EXPECT_EQ(
      qnsct1.qn,
      NormalQN1({QNCard("Sz", NormalQNVal(1))})
  );
  EXPECT_EQ(qnsct1.dim, 1);
  EXPECT_EQ(
      qnsct2.qn,
      NormalQN2({
          QNCard("Sz", NormalQNVal(-1)),
          QNCard("N",  NormalQNVal(1))})
  );
  EXPECT_EQ(qnsct2.dim, 2);
}


TEST_F(TestQNSector, Hashable) {
  std::hash<int> int_hasher;
  EXPECT_EQ(qnsct1_default.Hash(), (NormalQN1().Hash())^int_hasher(0));
  EXPECT_EQ(
      qnsct1.Hash(),
      (NormalQN1({QNCard("Sz", NormalQNVal(1))}).Hash())^int_hasher(1)
  );
  EXPECT_EQ(
      qnsct2.Hash(),
      (NormalQN2({
          QNCard("Sz", NormalQNVal(-1)),
          QNCard("N",  NormalQNVal(1))}).Hash())^int_hasher(2)
  );
}


TEST_F(TestQNSector, Equivalent) {
  EXPECT_TRUE(qnsct1_default == qnsct1_default);
  EXPECT_TRUE(qnsct1 == qnsct1);
  EXPECT_TRUE(qnsct2 == qnsct2);
  EXPECT_TRUE(qnsct1_default != qnsct1);
  /* TODO: I don't know whether it is useful to compare QNSector with different types. */
  //EXPECT_TRUE(qnsct1 != qnsct2);
}


template <typename QNSctT>
void RunTestQNSectorFileIOCase(const QNSctT &qnsct) {
  std::string file = "test.qnsct";
  std::ofstream out(file, std::ofstream::binary);
  out << qnsct;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNSctT qnsct_cpy;
  in >> qnsct_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qnsct_cpy, qnsct);
}


TEST_F(TestQNSector, FileIO) {
  RunTestQNSectorFileIOCase(qnsct1_default);
  RunTestQNSectorFileIOCase(qnsct1);
  RunTestQNSectorFileIOCase(qnsct2);
}
