# Source Code for Computing and Compressing Electron Repulsion Integrals on FPGAs

This repository contains the FPGA kernels for computing and compressing electron repulsion integrals, citation:

---
Wu, Xin; Kenter, Tobias; Schade, Robert; Kühne, Thomas D.; Plessl, Christian (2023):  
**Computing and Compressing Electron Repulsion Integrals on FPGAs.**  
In Proc. IEEE International Symposium on Field-Programmable Custom Computing Machines (FCCM), 2023. To appear.

---

## How to build the FPGA kernels?

```bash
dpcpp -DL_A=${am_a} -DL_B=${am_b} -DL_C=${am_c} -DL_D=${am_d}   \
      -DNQTBT=${nqtbt} -DNPRIC=${npric} -DNBITS=${nbits}        \
      -std=c++2b -Wall -Wextra -Wpedantic -Werror -fPIC         \
      -fintelfpga -O3 -qactypes                                 \
      -c eri-fpga.cpp -o eri-fpga.o
dpcpp -fPIC -shared -fintelfpga -O3 eri-fpga.o                  \
      -Xshardware -Xsboard=${FPGA_BOARD} -Xsparallel=16 -Xsv    \
      -Xsffp-reassociate -Xsffp-contract=fast                   \
      -o eri-fpga.so
```

where

- `am_a`: the angular momentum for Gaussian-type orbital a
- `am_b`: the angular momentum for Gaussian-type orbital b
- `am_c`: the angular momentum for Gaussian-type orbital c
- `am_d`: the angular momentum for Gaussian-type orbital d
- `nqtbt`: the number of quartets per batch execution
- `npric`: the number of private copies for data in FPGA local memory
- `nbits`: the bitwidth for the compression of electron repulsion integrals
- `FPGA_BOARD`: the FPGA board specification

