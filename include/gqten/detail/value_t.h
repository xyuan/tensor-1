// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-12 20:58
* 
* Description: GraceQ/tensor project. Type definitions used by this library.
*/
#ifndef GQTEN_DETAIL_VALUE_T_H
#define GQTEN_DETAIL_VALUE_T_H


#include <complex>

#include "gqten/detail/fwd_dcl.h"


namespace gqten {


// Public using types
using GQTEN_Double = double;
using GQTEN_Complex = std::complex<GQTEN_Double>;


using DGQTensor = GQTensor<GQTEN_Double>;
using ZGQTensor = GQTensor<GQTEN_Complex>;


// Intra using types
using QNSectorDimT_ = unsigned int;
} /* gqten */ 


// Redefine the complex number type for MKL
#define MKL_Complex16 gqten::GQTEN_Complex


#endif /* ifndef GQTEN_DETAIL_VALUE_T_H */
