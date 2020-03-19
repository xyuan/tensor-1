// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-03-18 20:53
* 
* Description: GraceQ/tensor project. Implementation details about tensor index.
*/
#include "gqten/gqten.h"

#include <iostream>     // istream, ostream


namespace gqten {


template <typename QNT>
InterOffsetQnsct<QNT> Index<QNT>::CoorInterOffsetAndQnsct(
    const long coor) const {
  long inter_offset = 0;
  for (auto &qnsct : QNSctSetT::qnscts) {
    long temp_inter_offset = inter_offset + qnsct.dim;
    if (temp_inter_offset > coor) {
      return InterOffsetQnsct<QNT>(inter_offset, qnsct);
    } else if (temp_inter_offset <= coor) {
      inter_offset = temp_inter_offset;
    }
  }
}

template <typename QNT>
void Index<QNT>::StreamRead(std::istream &is) {
  long qnscts_num;
  is >> qnscts_num;
  QNSctSetT::qnscts = QNSectorVec<QNT>(qnscts_num);
  for (auto &qnsct : QNSctSetT::qnscts) { is >> qnsct; }
  is >> dim >> dir;
  // Deal with empty tag, where will be '\n\n'.
  char next1_ch, next2_ch;
  is.get(next1_ch);
  is.get(next2_ch);
  if (next2_ch != '\n') {
    is.putback(next2_ch);
    is.putback(next1_ch);
    is >> tag;
  }
}


template <typename QNT>
void Index<QNT>::StreamWrite(std::ostream &os) const {
  long qnscts_num = QNSctSetT::qnscts.size();
  os << qnscts_num << std::endl;
  for (auto &qnsct : QNSctSetT::qnscts) { os << qnsct; }
  os << dim << std::endl;
  os << dir << std::endl;
  os << tag << std::endl;
}


template <typename IndexT>
IndexT InverseIndex(const IndexT &idx) {
  IndexT inversed_idx = idx;
  inversed_idx.Dag();
  return inversed_idx;
}
} /* gqten */ 
