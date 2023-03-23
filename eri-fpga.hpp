/*****************************************************************************************
 *                                                                                       *
 * FPGA kernels for computing and compressing electron repulsion integrals, please cite: *
 *                                                                                       *
 * Wu, Xin; Kenter, Tobias; Schade, Robert; Kühne, Thomas D.; Plessl, Christian (2023)   *
 * Computing and Compressing Electron Repulsion Integrals on FPGAs                       *
 * IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM)   *
 *                                                                                       *
 *****************************************************************************************/
#ifndef RYS4COMPUTE_HPP
#define RYS4COMPUTE_HPP 
#include <array>
#include <vector>
#include <CL/sycl.hpp>
#include <sycl/ext/intel/fpga_extensions.hpp>
#include <sycl/ext/intel/ac_types/ac_int.hpp>
using i_subq_t = unsigned char;
using i_quar_t = ac_int<14, false>;
using i_mulq_t = unsigned int;
constexpr i_quar_t k512 { 512 };
using MyInt = ac_int<NBITS, true>;
using MyInt512 = ac_int<k512, false>;
constexpr i_subq_t nxyz { 3 };
constexpr i_subq_t vier { 4 };
constexpr i_mulq_t num_qt_batch { NQTBT };
template<class T>
constexpr T sml_pow_of_2(const T n)
{
  T i { n };
  --i;
  i |= i >> 001;
  i |= i >> 002;
  i |= i >> 004;
  i |= i >> 010;
  return ++i ? i : 1;
}
template<class Real, unsigned short int abcd> class R4Compute;
template<class Real, unsigned char la,
                     unsigned char lb,
                     unsigned char lc,
                     unsigned char ld>
void Rys4Compute(sycl::queue& my_queue,
                 Real* abcd_dd,
                 Real* rtwt_dd,
                 MyInt512* cERI_dd,
                 Real* epsi_dd,
                 const std::vector<sycl::event>& prev_evt,
                                   sycl::event& evt_rys4)
{
  constexpr i_subq_t
  dim_blk { 16 },
  ls { la + lb + lc + ld },
  nORD {(ls >> 1) + 1 },
  ln { la + lb },
  lena { la + 1 },
  lenb { lb + 1 },
  lenn { ln + 1 },
  lm { lc + ld },
  lenc { lc + 1 },
  lend { ld + 1 },
  lenm { lm + 1 },
  nga {(la + 1) * (la + 2) >> 1 },
  ngb {(lb + 1) * (lb + 2) >> 1 },
  ngc {(lc + 1) * (lc + 2) >> 1 },
  ngd {(ld + 1) * (ld + 2) >> 1 },
  ngdc { ngd * ngc },
  ngba { ngb * nga },
  ngcba{ ngc * ngb * nga < 256 ?
         ngc * ngb * nga : 128 },
  dim_nxyz { sml_pow_of_2<i_subq_t>(nxyz) },
  dim_nORD { sml_pow_of_2<i_subq_t>(nORD) },
  dim_lend { sml_pow_of_2<i_subq_t>(lend) },
  dim_lenc { sml_pow_of_2<i_subq_t>(lenc) },
  dim_lenb { sml_pow_of_2<i_subq_t>(lenb) },
  dim_lena { sml_pow_of_2<i_subq_t>(lena) },
  dim_ngba { sml_pow_of_2<i_subq_t>(ngba) },
  ncin { k512 / NBITS };
  constexpr i_quar_t
  nERI { ngdc * ngba },
  n512 {(nERI + ncin - 1) / ncin };
  constexpr Real
  svalue { static_cast<Real>((0b1UL << (NBITS - 1)) - 1) },
  invt_s { 1.0 / svalue };
  constexpr i_quar_t neri_reg { 108 };
  evt_rys4 = my_queue.single_task
  <R4Compute<Real, 1000 * la + 100 * lb + 10 * lc + ld>>(prev_evt,[=]()
  [[intel::kernel_args_restrict]]
  {
    sycl::device_ptr<Real> abcd_aa { abcd_dd },
                               rtwt_aa { rtwt_dd };
    sycl::device_ptr<MyInt512> cERI_aa { cERI_dd };
    sycl::device_ptr<Real> epsi_aa { epsi_dd };
    [[intel::max_concurrency(NPRIC)]]
    for (i_mulq_t idx_quartet { 0 };
                  idx_quartet < num_qt_batch;
                  idx_quartet++) {
      [[intel::fpga_register]] Real ptom[5][nxyz],
                                    zt, et, ztet ,
                                    rtwt[dim_blk];
      {
      auto idx_base_abcd { idx_quartet* dim_blk};
      [[intel::fpga_register]] Real atom[vier][4 ],
                                    pq [2 ][nxyz],
                                    zt1, et1;
#pragma unroll
      for (i_subq_t i { 0 }; i < vier; i++) {
        auto j { idx_base_abcd + (i << 2) };
        atom[i][0] = abcd_aa[j + 0];
        atom[i][1] = abcd_aa[j + 1];
        atom[i][2] = abcd_aa[j + 2];
        atom[i][3] = abcd_aa[j + 3];
      }
      zt = atom[0][3] + atom[1][3];
      et = atom[2][3] + atom[3][3];
      zt1 = 1.0 / zt ;
      et1 = 1.0 / et ;
      ztet = 1.0 /(zt + et);
#pragma unroll
      for (i_subq_t i { 0 }; i < nxyz; i++) {
        ptom[0][i] = atom[0][i] - atom[1][i];
        ptom[1][i] = atom[2][i] - atom[3][i];
        pq[0][i] =(atom[0][3] * atom[0][i] + atom[1][3] * atom[1][i]) * zt1;
        pq[1][i] =(atom[2][3] * atom[2][i] + atom[3][3] * atom[3][i]) * et1;
        ptom[2][i] = pq[0][i] - atom[0][i];
        ptom[3][i] = pq[1][i] - atom[2][i];
        ptom[4][i] = pq[0][i] - pq [1][i];
      }
#pragma unroll
      for (i_subq_t i { 0 }; i < dim_blk; i++) {
        rtwt[i] = rtwt_aa[idx_base_abcd + i];
      }
      }
      [[intel::fpga_register]]
      Real tv, tb_cci[3][nORD],
               tb_ccd[2][nORD][nxyz];
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        tv = rtwt[iORD] * ztet;
        tb_cci[0][iORD] = 0.5 * tv ;
        tb_cci[1][iORD] = (0.5 - 0.5 * tv * zt) / et ;
        tb_cci[2][iORD] = (0.5 - 0.5 * tv * et) / zt ;
        tb_ccd[0][iORD][0] = ptom[3][0] + zt * ptom[4][0] * tv;
        tb_ccd[1][iORD][0] = ptom[2][0] - et * ptom[4][0] * tv;
        tb_ccd[0][iORD][1] = ptom[3][1] + zt * ptom[4][1] * tv;
        tb_ccd[1][iORD][1] = ptom[2][1] - et * ptom[4][1] * tv;
        tb_ccd[0][iORD][2] = ptom[3][2] + zt * ptom[4][2] * tv;
        tb_ccd[1][iORD][2] = ptom[2][2] - et * ptom[4][2] * tv;
      }
      if constexpr (nERI <= neri_reg) {
        if constexpr (nxyz * nORD * lend <= n512 && lend > 1) {
          if constexpr (ngd * ngc <= n512 && ngc > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      [[intel::loop_coalesce(3)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngdc][ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      [[intel::loop_coalesce(3)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          }
        } else if constexpr (nxyz * nORD <= n512 && nORD > 1) {
          if constexpr (ngd * ngc <= n512 && ngc > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngdc][ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else if constexpr (ngd <= n512 && ngd > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
#pragma unroll
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        bmax = abs_x > bmax ? abs_x : bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          }
        } else if constexpr (nxyz <= n512) {
          if constexpr (ngd * ngc <= n512 && ngc > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngdc][ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else if constexpr (ngd <= n512 && ngd > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
#pragma unroll
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        bmax = abs_x > bmax ? abs_x : bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          }
        } else {
          if constexpr (ngd * ngc <= n512 && ngc > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
#pragma unroll
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngdc][ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else if constexpr (ngd <= n512 && ngd > 1) {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
#pragma unroll
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          } else {
      [[intel::fpga_register]]
      Real int4d_eriq[lend][lenc][lenb][lena][nxyz][nORD];
#pragma unroll
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
#pragma unroll
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_register]]
      Real local_eriq[ngd][ngcba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
#pragma unroll
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
#pragma unroll
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd][igc * ngba + igb * nga + iga] = x;
        abs_x = std::fabs(x);
        bmax = abs_x > bmax ? abs_x : bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngcba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngd ) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngd ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
          }
        }
      } else {
        if constexpr (nxyz * nORD * lend <= n512 + 8 && lend > 1) {
      [[intel::fpga_memory,
        intel::private_copies(NPRIC),
        intel::max_replicates(1),
        intel::bankwidth(sizeof(Real)* dim_nORD),
        intel::numbanks( dim_lenc* dim_lenb* dim_lena* dim_nxyz)]]
      Real int4d_eriq[lend][dim_lenc][dim_lenb][dim_lena][dim_nxyz][dim_nORD];
      [[intel::loop_coalesce(3)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_memory,
        intel::private_copies(NPRIC),
        intel::max_replicates(ncin),
        intel::bankwidth(sizeof(Real)),
        intel::numbanks( dim_ngba)]]
      Real local_eriq[ngdc][dim_ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
        } else {
      [[intel::fpga_memory,
        intel::private_copies(NPRIC),
        intel::max_replicates(1),
        intel::bankwidth(sizeof(Real)* dim_nORD),
        intel::numbanks(dim_lend* dim_lenc* dim_lenb* dim_lena* dim_nxyz)]]
      Real int4d_eriq[ dim_lend][dim_lenc][dim_lenb][dim_lena][dim_nxyz][dim_nORD];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t ixyz { 0 }; ixyz < nxyz; ++ixyz) {
      for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
        [[intel::fpga_register]] Real int4d_dmbn[lend][lenm][lenb][lenn];
        int4d_dmbn[0][0][0][0] = rtwt[8 + iORD];
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
          Real t = idxn ?
          idxn * tb_cci[2][iORD] * int4d_dmbn[0][0][0][idxn - 1] : 0.0;
          int4d_dmbn[0][0][0][idxn + 1] = t + tb_ccd[1][iORD][ixyz] *
          int4d_dmbn[0][0][0][idxn ];
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm; ++idxm) {
          Real t = idxm ?
          idxm * tb_cci[1][iORD] * int4d_dmbn[0][idxm - 1][0][0] : 0.0;
          int4d_dmbn[0][idxm + 1][0][0] = t + tb_ccd[0][iORD][ixyz] *
          int4d_dmbn[0][idxm ][0][0];
#pragma unroll
          for (i_subq_t idxn { 0 }; idxn < ln; ++idxn) {
            Real t = idxn ?
            idxn * tb_cci[2][iORD] * int4d_dmbn[0][idxm + 1][0][idxn - 1] : 0.0;
            int4d_dmbn[0][idxm + 1][0][idxn + 1] = t
            + (idxm + 1) * tb_cci[0][iORD] * int4d_dmbn[0][idxm ][0][idxn]
            + tb_ccd[1][iORD][ixyz] * int4d_dmbn[0][idxm + 1][0][idxn];
          }
        }
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lenm ; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxn { 0 }; idxn < ln - idxb; ++idxn) {
          int4d_dmbn[0][idxm][idxb + 1][idxn ] =
          int4d_dmbn[0][idxm][idxb ][idxn + 1] +
          int4d_dmbn[0][idxm][idxb ][idxn ] * ptom[0][ixyz];
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < ld ; ++idxd) {
#pragma unroll
        for (i_subq_t idxm { 0 }; idxm < lm - idxd; ++idxm) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb ; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena ; ++idxa) {
          int4d_dmbn[idxd + 1][idxm ][idxb][idxa] =
          int4d_dmbn[idxd ][idxm + 1][idxb][idxa] +
          int4d_dmbn[idxd ][idxm ][idxb][idxa] * ptom[1][ixyz];
        }
        }
        }
        }
#pragma unroll
        for (i_subq_t idxd { 0 }; idxd < lend; ++idxd) {
#pragma unroll
        for (i_subq_t idxc { 0 }; idxc < lenc; ++idxc) {
#pragma unroll
        for (i_subq_t idxb { 0 }; idxb < lenb; ++idxb) {
#pragma unroll
        for (i_subq_t idxa { 0 }; idxa < lena; ++idxa) {
          int4d_eriq[idxd][idxc][idxb][idxa][ixyz][iORD] =
          int4d_dmbn[idxd][idxc][idxb][idxa];
        }
        }
        }
        }
      }
      }
      Real bmax;
      bmax = 0.0;
      [[intel::fpga_memory,
        intel::private_copies(NPRIC),
        intel::max_replicates(ncin),
        intel::bankwidth(sizeof(Real)),
        intel::numbanks( dim_ngba)]]
      Real local_eriq[ngdc][dim_ngba];
      i_subq_t i_d { 0 }, j_d { 0 }, k_d[nxyz];
      [[intel::loop_coalesce(2)]]
      for (i_subq_t igd { 0 }; igd < ngd; ++igd) {
        k_d[0] = ld - i_d;
        k_d[1] = i_d - j_d;
        k_d[2] = j_d;
      i_subq_t i_c { 0 }, j_c { 0 }, k_c[nxyz];
      for (i_subq_t igc { 0 }; igc < ngc; ++igc) {
        k_c[0] = lc - i_c;
        k_c[1] = i_c - j_c;
        k_c[2] = j_c;
        Real local_bmax;
        local_bmax = 0.0;
      i_subq_t i_b { 0 }, j_b { 0 }, k_b[nxyz];
#pragma unroll
      for (i_subq_t igb { 0 }; igb < ngb; ++igb) {
        k_b[0] = lb - i_b;
        k_b[1] = i_b - j_b;
        k_b[2] = j_b;
      i_subq_t i_a { 0 }, j_a { 0 }, k_a[nxyz];
#pragma unroll
      for (i_subq_t iga { 0 }; iga < nga; ++iga) {
        k_a[0] = la - i_a;
        k_a[1] = i_a - j_a;
        k_a[2] = j_a;
        Real x, abs_x;
        x = 0.0;
#pragma unroll
        for (i_subq_t iORD { 0 }; iORD < nORD; ++iORD) {
          x += int4d_eriq[k_d[0]][k_c[0]][k_b[0]][k_a[0]][0][iORD]
            * int4d_eriq[k_d[1]][k_c[1]][k_b[1]][k_a[1]][1][iORD]
            * int4d_eriq[k_d[2]][k_c[2]][k_b[2]][k_a[2]][2][iORD];
        }
        j_a++;
        local_eriq[igd * ngc + igc][igb * nga + iga] = x;
        abs_x = std::fabs(x);
        local_bmax = abs_x > local_bmax ? abs_x : local_bmax;
        if (j_a > i_a) {
            i_a++;
            j_a = 0;
        }
      }
        j_b++;
        if (j_b > i_b) {
            i_b++;
            j_b = 0;
        }
      }
        j_c++;
        bmax = bmax > local_bmax ? bmax : local_bmax;
        if (j_c > i_c) {
            i_c++;
            j_c = 0;
        }
      }
        j_d++;
        if (j_d > i_d) {
            i_d++;
            j_d = 0;
        }
      }
      epsi_aa[idx_quartet] = invt_s * bmax ;
      Real one_ov_epsilon { svalue / bmax };
      constexpr i_subq_t
      lUrL { ngba },
      nUrL {(ncin + lUrL - 1) / lUrL },
      ltmp { nUrL * lUrL },
      ltot { ncin - 1 + ltmp };
      [[intel::fpga_register]] std::array<MyInt, ncin - 1> rest;
      [[intel::fpga_register]] std::array<MyInt, ltmp > temp;
      i_subq_t next_ba { 1 },
               igdc { 0 },
               start_p { ncin - 1 };
      for (i_quar_t i512 { 0 };
                    i512 < n512;
                    i512++) {
      if (next_ba && igdc < ngdc) {
#pragma unroll
        for (i_subq_t jgba { 0 }; jgba < nUrL; jgba++) {
#pragma unroll
        for (i_subq_t igba { 0 }; igba < lUrL; igba++) {
          Real t;
          t = igdc + jgba < ngdc ?
              local_eriq[igdc + jgba][igba] * one_ov_epsilon :
              0.0;
          t = t > 0.0 ? t + 0.5 : t - 0.5;
          temp[jgba * lUrL + igba] = static_cast<MyInt>(t);
        }
        }
        next_ba = 0;
        igdc += nUrL;
      }
      MyInt512 cbuff;
#pragma unroll
      for (i_subq_t icin { 0 };
                    icin < ncin;
                    icin++) {
        i_subq_t j;
        j = start_p + icin;
        j < ncin - 1 ? cbuff.set_slc(icin * NBITS, rest[j ]):
                       cbuff.set_slc(icin * NBITS, temp[j - (ncin - 1)]);
      }
      cERI_aa[idx_quartet * n512 + i512] = cbuff;
      start_p += ncin;
      if (start_p + ncin > ltot) {
        start_p -= ltmp;
        next_ba = 1;
#pragma unroll
        for (i_subq_t k { 0 }; k < ncin - 1; k++) {
          rest[k] = temp[ltmp - ncin + 1 + k];
        }
      }
      }
        }
      }
    }
  });
}
#endif
