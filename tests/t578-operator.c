/// @file
/// Test linear element assembly of non-symmetric mass matrix operator (multi-component) (see t566)
/// \test Test linear element assembly of non-symmetric mass matrix operator (multi-component) (see t566)
#include "t566-operator.h"

#include <ceed.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Check that the output of an operator applied with its element matrices matches the output without them
static void CheckApply(CeedOperator op, CeedOperator op_elem_mats, CeedVector u, CeedVector v, CeedVector v_elem_mats, const char *label) {
  CeedSize          length;
  CeedScalar        max_value = 0.0;
  const CeedScalar *v_array, *v_elem_mats_array;

  CeedOperatorApply(op, u, v, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_elem_mats, u, v_elem_mats, CEED_REQUEST_IMMEDIATE);
  CeedVectorGetLength(v, &length);
  CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
  CeedVectorGetArrayRead(v_elem_mats, CEED_MEM_HOST, &v_elem_mats_array);
  for (CeedSize i = 0; i < length; i++) max_value = fmax(max_value, fabs(v_array[i]));
  for (CeedSize i = 0; i < length; i++) {
    if (fabs(v_array[i] - v_elem_mats_array[i]) > 100. * CEED_EPSILON * max_value) {
      // LCOV_EXCL_START
      printf("%s: [%" CeedSize_FMT "] Error in application with element matrices: %f != %f\n", label, i, v_elem_mats_array[i], v_array[i]);
      // LCOV_EXCL_STOP
    }
  }
  CeedVectorRestoreArrayRead(v, &v_array);
  CeedVectorRestoreArrayRead(v_elem_mats, &v_elem_mats_array);
}

int main(int argc, char **argv) {
  Ceed                ceed;
  CeedElemRestriction elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis           basis_x, basis_u;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass, op_mass_elem_mats;
  CeedVector          q_data, x, u, v, v_elem_mats;
  CeedInt             p = 3, q = 3, dim = 2, num_comp = 2;
  CeedInt             n_x = 1, n_y = 1;
  CeedInt             num_elem = n_x * n_y;
  CeedInt             num_dofs = (n_x * (p - 1) + 1) * (n_y * (p - 1) + 1), num_qpts = num_elem * q * q;
  CeedInt             ind_x[num_elem * p * p];

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_x * (p - 1) + 1; i++) {
      for (CeedInt j = 0; j < n_y * (p - 1) + 1; j++) {
        x_array[i + j * (n_x * 2 + 1) + 0 * num_dofs] = (CeedScalar)i / (n_x * (p - 1));
        x_array[i + j * (n_x * 2 + 1) + 1 * num_dofs] = (CeedScalar)j / (n_y * (p - 1));
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_comp * num_dofs, &u);
  CeedVectorCreate(ceed, num_comp * num_dofs, &v);
  CeedVectorCreate(ceed, num_comp * num_dofs, &v_elem_mats);
  CeedVectorCreate(ceed, num_qpts, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;

    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p - 1) + row * (n_x * (p - 1) + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) ind_x[p * (p * i + k) + j] = offset + k * p + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, num_comp, num_dofs, num_comp * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_x,
                            &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q * num_elem, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, num_comp, p, q, CEED_GAUSS, &basis_u);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", num_comp, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", num_comp, CEED_EVAL_INTERP);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_elem_mats);
  CeedOperatorSetField(op_mass_elem_mats, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_elem_mats, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_elem_mats, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  // Apply Setup Operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);

  // Input
  {
    CeedSize    length;
    CeedScalar *u_array;

    CeedVectorGetLength(u, &length);
    CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
    for (CeedSize i = 0; i < length; i++) u_array[i] = sin(1.0 + i);
    CeedVectorRestoreArray(u, &u_array);
  }

  // Apply with element matrices
  CeedOperatorSetLinearElementAssembly(op_mass_elem_mats, true);
  CheckApply(op_mass, op_mass_elem_mats, u, v, v_elem_mats, "Initial");

  // Update passive input
  CeedVectorScale(q_data, 0.5);
  CheckApply(op_mass, op_mass_elem_mats, u, v, v_elem_mats, "Passive input update");

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_x);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedVectorDestroy(&v_elem_mats);
  CeedOperatorDestroy(&op_mass_elem_mats);
  CeedDestroy(&ceed);
  return 0;
}
