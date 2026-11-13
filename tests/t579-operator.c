/// @file
/// Test linear element assembly of a non-square mass matrix operator (see t569)
/// \test Test linear element assembly of a non-square mass matrix operator (see t569)
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
  CeedElemRestriction elem_restriction_x, elem_restriction_q_data, elem_restriction_u_coarse, elem_restriction_u_fine;
  CeedBasis           basis_x, basis_u_coarse, basis_u_fine;
  CeedQFunction       qf_setup, qf_mass;
  CeedOperator        op_setup, op_mass, op_mass_elem_mats;
  CeedVector          q_data, x, u, v, v_elem_mats;
  CeedInt             num_elem = 15, p_coarse = 3, p_fine = 5, q = 8;
  CeedInt             num_dofs_x = num_elem + 1, num_dofs_u_coarse = num_elem * (p_coarse - 1) + 1, num_dofs_u_fine = num_elem * (p_fine - 1) + 1;
  CeedInt             ind_u_coarse[num_elem * p_coarse], ind_u_fine[num_elem * p_fine], ind_x[num_elem * 2];

  CeedInit(argv[1], &ceed);

  CeedVectorCreate(ceed, num_dofs_x, &x);
  {
    CeedScalar x_array[num_dofs_x];

    for (CeedInt i = 0; i < num_dofs_x; i++) x_array[i] = (CeedScalar)i / (num_dofs_x - 1);
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs_u_coarse, &u);
  CeedVectorCreate(ceed, num_dofs_u_fine, &v);
  CeedVectorCreate(ceed, num_dofs_u_fine, &v_elem_mats);
  CeedVectorCreate(ceed, num_elem * q, &q_data);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    ind_x[2 * i + 0] = i;
    ind_x[2 * i + 1] = i + 1;
  }
  CeedElemRestrictionCreate(ceed, num_elem, 2, 1, 1, num_dofs_x, CEED_MEM_HOST, CEED_USE_POINTER, ind_x, &elem_restriction_x);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_coarse; j++) {
      ind_u_coarse[p_coarse * i + j] = i * (p_coarse - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_coarse, 1, 1, num_dofs_u_coarse, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_coarse,
                            &elem_restriction_u_coarse);

  for (CeedInt i = 0; i < num_elem; i++) {
    for (CeedInt j = 0; j < p_fine; j++) {
      ind_u_fine[p_fine * i + j] = i * (p_fine - 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p_fine, 1, 1, num_dofs_u_fine, CEED_MEM_HOST, CEED_USE_POINTER, ind_u_fine, &elem_restriction_u_fine);

  CeedInt strides_q_data[3] = {1, q, q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q, 1, q * num_elem, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p_coarse, q, CEED_GAUSS, &basis_u_coarse);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, p_fine, q, CEED_GAUSS, &basis_u_fine);

  // QFunctions
  CeedQFunctionCreateInteriorByName(ceed, "Mass1DBuild", &qf_setup);
  CeedQFunctionCreateInteriorByName(ceed, "MassApply", &qf_mass);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);

  CeedOperatorSetField(op_setup, "weights", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass);
  CeedOperatorSetField(op_mass, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass, "u", elem_restriction_u_coarse, basis_u_coarse, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass, "v", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_elem_mats);
  CeedOperatorSetField(op_mass_elem_mats, "qdata", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_elem_mats, "u", elem_restriction_u_coarse, basis_u_coarse, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_elem_mats, "v", elem_restriction_u_fine, basis_u_fine, CEED_VECTOR_ACTIVE);

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
  CeedElemRestrictionDestroy(&elem_restriction_u_coarse);
  CeedElemRestrictionDestroy(&elem_restriction_u_fine);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_u_coarse);
  CeedBasisDestroy(&basis_u_fine);
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
