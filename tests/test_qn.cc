// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-24 21:32
* 
* Description: GraceQ/tensor project. Unit tests for quantum number object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"
#include "gqten/detail/vec_hash.h"

#include <string>
#include <fstream>

#include <cstdio>


using namespace gqten;


using QN0 = QN<>;
using NormalQN1 = QN<NormalQNVal>;
using NormalQN2 = QN<NormalQNVal, NormalQNVal>;


struct TestQN : public testing::Test {
  QN0 qn_default = QN0();
  NormalQN1 qn_u1_1 = NormalQN1({QNCard("Sz", NormalQNVal(0))});
  NormalQN1 qn_u1_2 = NormalQN1({QNCard("Sz", NormalQNVal(1))});
  NormalQN1 qn_u1_3 = NormalQN1({QNCard("Sz", NormalQNVal(-1))});
  NormalQN2 qn_u1_u1_1 = NormalQN2({QNCard("Sz", NormalQNVal(0)),
                                    QNCard("N",  NormalQNVal(0))});
  NormalQN2 qn_u1_u1_2 = NormalQN2({QNCard("Sz", NormalQNVal(1)),
                                    QNCard("N",  NormalQNVal(2))});
};


TEST_F(TestQN, Hashable) {
  EXPECT_EQ(qn_default.Hash(), 0);
  EXPECT_EQ(qn_u1_1.Hash(), VecStdTypeHasher(std::vector<long>{0}));
  EXPECT_EQ(qn_u1_2.Hash(), VecStdTypeHasher(std::vector<long>{1}));
  EXPECT_EQ(qn_u1_3.Hash(), VecStdTypeHasher(std::vector<long>{-1}));
  EXPECT_EQ(qn_u1_u1_1.Hash(), VecStdTypeHasher(std::vector<long>{0, 0}));
  EXPECT_EQ(qn_u1_u1_2.Hash(), VecStdTypeHasher(std::vector<long>{1, 2}));
}


TEST_F(TestQN, Equivalent) {
  EXPECT_TRUE(qn_default == QN0());
  EXPECT_TRUE(qn_u1_1 == qn_u1_1);
  EXPECT_TRUE(qn_u1_u1_1 == qn_u1_u1_1);
  EXPECT_TRUE(qn_u1_1 != qn_u1_2);
  /* TODO: I don't know whether it is useful to compare QN with different types. */
  //EXPECT_TRUE(qn_u1_1 != qn_u1_u1_2);
}


TEST_F(TestQN, Negtivation) {
  EXPECT_EQ(-qn_default, QN0());
  EXPECT_EQ(-qn_u1_1, NormalQN1({QNCard("Sz", NormalQNVal(0))}));
  EXPECT_EQ(-qn_u1_2, NormalQN1({QNCard("Sz", NormalQNVal(-1))}));
}


TEST_F(TestQN, Summation) {
  auto res0 = qn_default + qn_default;
  EXPECT_EQ(res0, QN0());
  auto res1 = qn_u1_2 + qn_u1_2;
  EXPECT_EQ(res1, NormalQN1({QNCard("Sz", NormalQNVal(2))}));
  auto res2 = qn_u1_u1_1 + qn_u1_u1_2;
  EXPECT_EQ(res2,
            NormalQN2({QNCard("Sz", NormalQNVal(1)),
                       QNCard("N", NormalQNVal(2))}));

}


TEST_F(TestQN, Subtraction) {
  auto res = qn_u1_1 - qn_u1_2;
  EXPECT_EQ(res, qn_u1_3);
}


template<typename QNT>
void RunTestQNFileIOCase(const QNT &qn) {
  std::string file = "test.qn";
  std::ofstream out(file, std::ofstream::binary);
  out << qn;
  out.close();
  std::ifstream in(file, std::ifstream::binary);
  QNT qn_cpy;
  in >> qn_cpy;
  in.close();
  std::remove(file.c_str());
  EXPECT_EQ(qn_cpy, qn);
}


TEST_F(TestQN, FileIO) {
  RunTestQNFileIOCase(qn_default);
  RunTestQNFileIOCase(qn_u1_1);
  RunTestQNFileIOCase(qn_u1_2);
  RunTestQNFileIOCase(qn_u1_3);
  RunTestQNFileIOCase(qn_u1_u1_1);
  RunTestQNFileIOCase(qn_u1_u1_2);
}
