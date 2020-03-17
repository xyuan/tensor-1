// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-26 14:54
* 
* Description: GraceQ/tensor project. Unit tests for QNSectorSet class.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"

#include <vector>


using namespace gqten;


using NormalQN1 = QN<NormalQNVal>;

using QNSctT = QNSector<NormalQN1>;

using QNSctSetT = QNSectorSet<NormalQN1>;

struct TestQNSectorSet : public testing::Test {
  QNSctSetT qnscts_default;

  QNSectorVec<NormalQN1> qnscts1 = {
      QNSctT(NormalQN1({QNCard("Sz", NormalQNVal(0))}), 1)
  };
  QNSctSetT qnscts_1sct = QNSctSetT(qnscts1);

  ConstQNSectorPtrVec<NormalQN1> pqnscts1 = {&qnscts1[0]};
  QNSctSetT qnscts_1sct_from_ptr = QNSctSetT(pqnscts1);

  QNSectorVec<NormalQN1> qnscts2 = {
      QNSctT(NormalQN1({QNCard("Sz", NormalQNVal(0))}), 1),
      QNSctT(NormalQN1({QNCard("Sz", NormalQNVal(1))}), 2)
  };
  QNSctSetT qnscts_2sct = QNSctSetT(qnscts2);
};


TEST_F(TestQNSectorSet, DataMembers) {
  QNSectorVec<NormalQN1> empty_qnscts;
  EXPECT_EQ(qnscts_default.qnscts, empty_qnscts);
  EXPECT_EQ(qnscts_1sct.qnscts, qnscts1);
  EXPECT_EQ(qnscts_1sct_from_ptr.qnscts, qnscts1);
  EXPECT_EQ(qnscts_2sct.qnscts, qnscts2);
}


TEST_F(TestQNSectorSet, Equivalent) {
  EXPECT_TRUE(qnscts_default == qnscts_default);
  EXPECT_TRUE(qnscts_1sct == qnscts_1sct);
  EXPECT_TRUE(qnscts_2sct == qnscts_2sct);
  EXPECT_TRUE(qnscts_1sct == qnscts_1sct_from_ptr);
  EXPECT_TRUE(qnscts_1sct != qnscts_2sct);
}
