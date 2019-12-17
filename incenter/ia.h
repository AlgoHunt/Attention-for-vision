#pragma once
#include "torch/extension.h"


#define AT_DISPATCH_FLOAT_TYPE(TYPE, NAME, ...)                                \
  [&] {                                                                        \
    const at::Type &the_type = TYPE;                                           \
    switch (the_type.scalarType()) {                                           \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Float, float, __VA_ARGS__)          \
    default:                                                                   \
      AT_ERROR(#NAME, " not implemented for '", the_type.toString(), "'");     \
    }                                                                          \
  }()


