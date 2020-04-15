// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-09-17 13:21
* 
* Description: GraceQ/tensor project. Forward declarations for implementing tensor contraction.
*/
#ifndef GQTEN_DETAIL_TEN_CTRCT_FWD_H
#define GQTEN_DETAIL_TEN_CTRCT_FWD_H


#include <vector>

#include "gqten/detail/fwd_dcl.h"


namespace gqten {


template <typename QNT, typename TenElemType>
void InitCtrctedTen(
    const GQTensor<QNT, TenElemType> *, const GQTensor<QNT, TenElemType> *,
    const std::vector<long> &, const std::vector<long> &,
    GQTensor<QNT, TenElemType> *
);

template <typename QNT, typename TenElemType>
void WrapCtrctBlocks(
    std::vector<QNBlock<QNT, TenElemType> *> &,
    GQTensor<QNT, TenElemType> *
);


template <typename QNT, typename TenElemType>
std::vector<QNBlock<QNT, TenElemType> *> MergeCtrctBlks(
    const std::vector<QNBlock<QNT, TenElemType> *> &
);

template <typename QNT, typename TenElemType>
std::vector<const QNSector<QNT> *> GetPNewBlkQNScts(
    const QNBlock<QNT, TenElemType> *, const QNBlock<QNT, TenElemType> *,
    const std::vector<long> &, const std::vector<long> &
);

template <typename TenElemType>
bool CtrctTransChecker(
    const std::vector<long> &,
    const long,
    const char,
    std::vector<long> &
);

template <typename QNT, typename TenElemType>
std::vector<std::size_t> GenBlksPartHashTable(
    const std::vector<QNBlock<QNT, TenElemType> *> &, const std::vector<long> &
);

template <typename QNT, typename TenElemType>
std::vector<QNBlock<QNT, TenElemType> *> BlocksCtrctBatch(
    const std::vector<long> &, const std::vector<long> &,
    const TenElemType,
    const std::vector<QNBlock<QNT, TenElemType> *> &,
    const std::vector<QNBlock<QNT, TenElemType> *> &
);
} /* gqten */ 
#endif /* ifndef GQTEN_DETAIL_TEN_CTRCT_FWD_H */
