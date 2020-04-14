// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-08-08 17:07
* 
* Description: GraceQ/tensor project. Forward declaration for intra-used utility functions.
*/
#ifndef GQTEN_UTILS_H
#define GQTEN_UTILS_H


#include <vector>
#include <complex>

#include "gqten/detail/fwd_dcl.h"


namespace gqten {


std::vector<long> CalcMultiDimDataOffsets(const std::vector<long> &);

long MulToEnd(const std::vector<long> &, int);
} /* gqten */ 
#endif /* ifndef GQTEN_UTILS_H */
