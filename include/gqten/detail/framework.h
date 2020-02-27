// SPDX-License-Identifier: LGPL-3.0-only
/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2020-02-25 14:21
* 
* Description: GraceQ/tensor project. Basic framework.
*/
#ifndef GQTEN_DETAIL_FRAMEWORK_H
#define GQTEN_DETAIL_FRAMEWORK_H


#include <iostream>     // istream, ostream


namespace gqten {


// Abstract base class for streamable object
class Streamable {
public:
  Streamable(void) = default;
  virtual ~Streamable(void) = default;

  virtual void StreamRead(std::istream &) = 0;
  virtual void StreamWrite(std::ostream &) const = 0;
};


// Overload I/O operators for streamable object
inline
std::istream &operator>>(std::istream &is, Streamable &obj) {
  obj.StreamRead(is);
  return is;
}


inline
std::ostream &operator<<(std::ostream &os, const Streamable &obj) {
  obj.StreamWrite(os);
  return os;
}


// Abstract base class for hashable object
class Hashable {
public:
  Hashable(void) = default;
  virtual ~Hashable(void) = default;

  virtual std::size_t Hash(void) const = 0;
};


inline
std::size_t Hash(const Hashable &obj) { return obj.Hash(); }


} /* gqten */ 


#endif /* ifndef GQTEN_DETAIL_FRAMEWORK_H */
