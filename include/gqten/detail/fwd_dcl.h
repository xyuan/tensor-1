// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-17 09:52
* 
* Description: GraceQ/tensor project. Forward declarations.
*/
#ifndef GQTEN_DETAIL_FWD_DCL_H
#define GQTEN_DETAIL_FWD_DCL_H


#include <fstream>


namespace gqten {


template <typename ...>
class QN;


template <typename>
class QNSector;


template <typename>
class QNSectorSet;


template <typename>
class Index;


// Directions of the index
#define NDIR "NDIR"
#define IN "IN"
#define OUT "OUT"


template <typename, typename>
class QNBlock;


template <typename, typename>
class GQTensor;
} /* gqten */ 
#endif /* ifndef GQTEN_DETAIL_FWD_DCL_H */
