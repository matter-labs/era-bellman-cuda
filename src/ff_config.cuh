// finite field definitions

#pragma once

#include "ff_storage.cuh"

struct ff_config_p {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
  // modulus = 21888242871839275222246405745257275088696311157297823662689037894645226208583
  static constexpr ff_storage<limbs_count> modulus = {0xd87cfd47, 0x3c208c16, 0x6871ca8d, 0x97816a91, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
  // modulus*2 = 43776485743678550444492811490514550177392622314595647325378075789290452417166
  static constexpr ff_storage<limbs_count> modulus_2 = {0xb0f9fa8e, 0x7841182d, 0xd0e3951a, 0x2f02d522, 0x0302b0bb, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
  // modulus*4 = 87552971487357100888985622981029100354785244629191294650756151578580904834332
  static constexpr ff_storage<limbs_count> modulus_4 = {0x61f3f51c, 0xf082305b, 0xa1c72a34, 0x5e05aa45, 0x06056176, 0xe14116da, 0x84c680a6, 0xc19139cb};
  // modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared = {0x275d69b1, 0x3b5458a2, 0x09eac101, 0xa602072d, 0x6d96cadc, 0x4a50189c,
                                                                   0x7a1242c8, 0x04689e95, 0x34c6b38d, 0x26edfa5c, 0x16375606, 0xb00b8551,
                                                                   0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
  // 2*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {0x4ebad362, 0x76a8b144, 0x13d58202, 0x4c040e5a, 0xdb2d95b9, 0x94a03138,
                                                                     0xf4248590, 0x08d13d2a, 0x698d671a, 0x4ddbf4b8, 0x2c6eac0c, 0x60170aa2,
                                                                     0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
  // 4*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {0x9d75a6c4, 0xed516288, 0x27ab0404, 0x98081cb4, 0xb65b2b72, 0x29406271,
                                                                     0xe8490b21, 0x11a27a55, 0xd31ace34, 0x9bb7e970, 0x58dd5818, 0xc02e1544,
                                                                     0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
  // r2 = 3096616502983703923843567936837374451735540968419076528771170197431451843209
  static constexpr ff_storage<limbs_count> r2 = {0x538afa89, 0xf32cfc5b, 0xd44501fb, 0xb5e71911, 0x0a417ff6, 0x47ab1eff, 0xcab8351f, 0x06d89f71};
  // inv
  static constexpr uint32_t inv = 0xe4866389;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0xc58f0d9d, 0xd35d438d, 0xf5c70b3d, 0x0a78eb28, 0x7879462c, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
  static constexpr unsigned modulus_bits_count = 254;
};

// Can't make this a member of ff_config_p. nvcc does not allow __constant__ on members.
extern __device__ __constant__ uint32_t inv_p;

struct ff_config_q {
  // field structure size = 8 * 32 bit
  static constexpr unsigned limbs_count = 8;
  // modulus = 21888242871839275222246405745257275088548364400416034343698204186575808495617
  static constexpr ff_storage<limbs_count> modulus = {0xf0000001, 0x43e1f593, 0x79b97091, 0x2833e848, 0x8181585d, 0xb85045b6, 0xe131a029, 0x30644e72};
  // modulus*2 = 43776485743678550444492811490514550177096728800832068687396408373151616991234
  static constexpr ff_storage<limbs_count> modulus_2 = {0xe0000002, 0x87c3eb27, 0xf372e122, 0x5067d090, 0x0302b0ba, 0x70a08b6d, 0xc2634053, 0x60c89ce5};
  // modulus*4 = 87552971487357100888985622981029100354193457601664137374792816746303233982468
  static constexpr ff_storage<limbs_count> modulus_4 = {0xc0000004, 0x0f87d64f, 0xe6e5c245, 0xa0cfa121, 0x06056174, 0xe14116da, 0x84c680a6, 0xc19139cb};
  // modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared = {0xe0000001, 0x08c3eb27, 0xdcb34000, 0xc7f26223, 0x68c9bb7f, 0xffe9a62c,
                                                                   0xe821ddb0, 0xa6ce1975, 0x47b62fe7, 0x2c77527b, 0xd379d3df, 0x85f73bb0,
                                                                   0x0348d21c, 0x599a6f7c, 0x763cbf9c, 0x0925c4b8};
  // 2*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_2 = {0xc0000002, 0x1187d64f, 0xb9668000, 0x8fe4c447, 0xd19376ff, 0xffd34c58,
                                                                     0xd043bb61, 0x4d9c32eb, 0x8f6c5fcf, 0x58eea4f6, 0xa6f3a7be, 0x0bee7761,
                                                                     0x0691a439, 0xb334def8, 0xec797f38, 0x124b8970};
  // 4*modulus^2
  static constexpr ff_storage_wide<limbs_count> modulus_squared_4 = {0x80000004, 0x230fac9f, 0x72cd0000, 0x1fc9888f, 0xa326edff, 0xffa698b1,
                                                                     0xa08776c3, 0x9b3865d7, 0x1ed8bf9e, 0xb1dd49ed, 0x4de74f7c, 0x17dceec3,
                                                                     0x0d234872, 0x6669bdf0, 0xd8f2fe71, 0x249712e1};
  // r2 = 944936681149208446651664254269745548490766851729442924617792859073125903783
  static constexpr ff_storage<limbs_count> r2 = {0xae216da7, 0x1bb8e645, 0xe35c59e3, 0x53fe3ab1, 0x53bb8085, 0x8c49833d, 0x7f4e44a5, 0x0216d0b1};
  // inv
  static constexpr uint32_t inv = 0xefffffff;
  // 1 in montgomery form
  static constexpr ff_storage<limbs_count> one = {0x4ffffffb, 0xac96341c, 0x9f60cd29, 0x36fc7695, 0x7879462e, 0x666ea36f, 0x9a07df2f, 0x0e0a77c1};
  static constexpr unsigned modulus_bits_count = 254;
  // log2 of order of omega
  static constexpr unsigned omega_log_order = 28;
  // k=(modulus-1)/(2^omega_log_order) = (21888242871839275222246405745257275088548364400416034343698204186575808495617-1)/(2^28) =
  // 81540058820840996586704275553141814055101440848469862132140264610111
  // omega generator is 7
  static constexpr unsigned omega_generator = 7;
  // omega = generator^k mod P = 7^81540058820840996586704275553141814055101440848469862132140264610111 mod
  // 21888242871839275222246405745257275088548364400416034343698204186575808495617 =
  // 1748695177688661943023146337482803886740723238769601073607632802312037301404 =
  // omega in montgomery form
  static constexpr ff_storage<limbs_count> omega = {0xb639feb8, 0x9632c7c5, 0x0d0ff299, 0x985ce340, 0x01b0ecd8, 0xb2dd8800, 0x6d98ce29, 0x1d69070d};
  // inverse of 2 in montgomery form
  static constexpr ff_storage<limbs_count> two_inv = {0x1ffffffe, 0x783c14d8, 0x0c8d1edd, 0xaf982f6f, 0xfcfd4f45, 0x8f5f7492, 0x3d9cbfac, 0x1f37631a};
};

// Can't make this a member of ff_config_q. nvcc does not allow __constant__ on members.
extern __device__ __constant__ uint32_t inv_q;
