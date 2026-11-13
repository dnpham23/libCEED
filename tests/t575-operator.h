// Copyright (c) 2017-2026, Lawrence Livermore National Security, LLC and other CEED contributors.
// All Rights Reserved. See the top-level LICENSE and NOTICE files for details.
//
// SPDX-License-Identifier: BSD-2-Clause
//
// This file is part of CEED:  http://github.com/ceed

#include <ceed/types.h>

typedef struct {
  CeedScalar scale;
} ScaleContext;

CEED_QFUNCTION(setup)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *weight = in[0], *J = in[1];
  CeedScalar       *rho = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    rho[i] = weight[i] * (J[i + Q * 0] * J[i + Q * 3] - J[i + Q * 1] * J[i + Q * 2]);
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(scaled_mass)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const ScaleContext *context = (const ScaleContext *)ctx;
  const CeedScalar   *rho = in[0], *u = in[1];
  CeedScalar         *v = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = context->scale * rho[i] * u[i];
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(mass_two_inputs)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u_1 = in[1], *u_2 = in[2];
  CeedScalar       *v = out[0];
  for (CeedInt i = 0; i < Q; i++) {
    v[i] = rho[i] * (u_1[i] + u_2[i]);
  }
  return CEED_ERROR_SUCCESS;
}

CEED_QFUNCTION(mass_passive_output)(void *ctx, const CeedInt Q, const CeedScalar *const *in, CeedScalar *const *out) {
  const CeedScalar *rho = in[0], *u = in[1];
  CeedScalar       *v = out[0], *rho_copy = out[1];
  for (CeedInt i = 0; i < Q; i++) {
    v[i]        = rho[i] * u[i];
    rho_copy[i] = rho[i];
  }
  return CEED_ERROR_SUCCESS;
}
