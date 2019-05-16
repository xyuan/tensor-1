/*
* Author: Rongyang Sun <sun-rongyang@outlook.com>
* Creation Date: 2019-04-27 11:21
* 
* Description: GraceQ/tensor project. Calculate the hash value of a vector
*              whose item has a method called Hash().
*/
#ifndef GQTEN_VEC_HASH_H
#define GQTEN_VEC_HASH_H

#define _HASH_XXPRIME_1 ((size_t)11400714785074694791ULL)
#define _HASH_XXPRIME_2 ((size_t)14029467366897019727ULL)
#define _HASH_XXPRIME_5 ((size_t)2870177450012600261ULL)
#define _HASH_XXROTATE(x) ((x << 31) | (x >> 33))

#include <vector>
#include <string>


namespace gqten {


template<typename T>
size_t VecHasher(const std::vector<T> &vec) {
  size_t len = vec.size();
  size_t hash_val = _HASH_XXPRIME_5;
  for (auto &item : vec) {
    size_t item_hash_val = item.Hash();
    hash_val += item_hash_val * _HASH_XXPRIME_2;
    hash_val = _HASH_XXROTATE(hash_val);
    hash_val *= _HASH_XXPRIME_1;
  }
  hash_val += len ^ _HASH_XXPRIME_5;
  return hash_val;
}


// Helper classes.
class HashableString {
public:
  HashableString(void) : str_("") {}
  HashableString(const std::string &nstr) : str_(nstr) {} 
  HashableString(const long num) : str_(std::to_string(num)) {}

  std::size_t Hash(void) const { return strhasher_(str_); }

private:
  std::string str_;
  static std::hash<std::string> strhasher_;
};
} /* gqten */ 
#endif /* ifndef GQTEN_VEC_HASH_H */
