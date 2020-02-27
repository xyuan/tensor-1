// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-02-25 17:29
* 
* Description: GraceQ/tensor project. Unit tests for quantum number value object.
*/
#include "gtest/gtest.h"
#include "gqten/gqten.h"


using namespace gqten;


struct TestAbelQNVal : public testing::Test {
  AbelQNVal abelqnval0 = AbelQNVal();
  AbelQNVal abelqnval1 = AbelQNVal(1);
  AbelQNVal abelqnval2 = AbelQNVal(2);
  AbelQNVal abelqnvalm1 = AbelQNVal(-1);
};


TEST_F(TestAbelQNVal, Hashable) {
  std::hash<long> Hasher;
  EXPECT_EQ(Hash(abelqnval0), Hasher(0));
  EXPECT_EQ(Hash(abelqnval1), Hasher(1));
  EXPECT_EQ(Hash(abelqnval2), Hasher(2));
  EXPECT_EQ(Hash(abelqnvalm1), Hasher(-1));
}
