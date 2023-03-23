#include "eri-fpga.hpp"
#ifndef L_A
#define L_A 0
#endif
#ifndef L_B
#define L_B 0
#endif
#ifndef L_C
#define L_C 0
#endif
#ifndef L_D
#define L_D 0
#endif
#ifndef NQTBT
#define NQTBT (262144)
#endif
#ifndef NPRIC
#define NPRIC (8)
#endif
#ifndef NBITS
#define NBITS (16)
#endif
#define CAT(a, b) a##b
#define GENNAME(p, x) CAT(p, x)
#define F_FLOAT_DEF(q, abcd, rtwt, cERI, epsi, p, e) extern "C" void          \
  GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(GENNAME(Rys4Compute_float_, L_A), L_B), L_C), L_D), _), NQTBT), _), NPRIC), _), NBITS) ( \
  sycl::queue& q,                                                             \
  float*    abcd, float* rtwt,                                                \
  MyInt512* cERI, float* epsi,                                                \
  const std::vector<sycl::event>& p, sycl::event& e)                          \
  { Rys4Compute<float, L_A, L_B, L_C, L_D>(q, abcd, rtwt, cERI, epsi, p, e); };
// C wrapper function
F_FLOAT_DEF(my_queue, abcd_dd, rtwt_dd, cERI_dd, epsi_dd, prev_evt, evt_rys4)
