/// @file
/// Test linear element assembly of mass matrix operator with a context, of composite and quadrature point operators, and of unsupported operators
/// \test Test linear element assembly of mass matrix operator with a context, of composite and quadrature point operators, and of unsupported operators
#include "t575-operator.h"

#include <ceed.h>
#include <math.h>
#include <stddef.h>
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
  Ceed                  ceed;
  CeedElemRestriction   elem_restriction_x, elem_restriction_u, elem_restriction_q_data;
  CeedBasis             basis_x, basis_u, basis_u_copy;
  CeedQFunction         qf_setup, qf_mass, qf_mass_writable, qf_mass_two_inputs, qf_mass_passive_output, qf_scale;
  CeedQFunctionContext  ctx_mass;
  CeedOperator          op_setup, op_mass, op_mass_elem_mats, op_mass_2, op_mass_elem_mats_2, op_composite, op_composite_elem_mats;
  CeedOperator          op_mass_writable, op_mass_two_inputs, op_mass_passive_output, op_scale, op_scale_elem_mats;
  CeedContextFieldLabel scale_label;
  CeedVector            q_data, q_data_copy, x, u, v, v_elem_mats, u_q, v_q, v_q_elem_mats;
  CeedInt               p = 3, q = 4, dim = 2;
  CeedInt               n_x = 3, n_y = 2;
  CeedInt               num_elem = n_x * n_y;
  CeedInt               num_dofs = (n_x * (p - 1) + 1) * (n_y * (p - 1) + 1), num_qpts = num_elem * q * q;
  CeedInt               ind_u[num_elem * p * p];
  ScaleContext          scale_context = {.scale = 1.5};

  CeedInit(argv[1], &ceed);

  // Vectors
  CeedVectorCreate(ceed, dim * num_dofs, &x);
  {
    CeedScalar x_array[dim * num_dofs];

    for (CeedInt i = 0; i < n_x * (p - 1) + 1; i++) {
      for (CeedInt j = 0; j < n_y * (p - 1) + 1; j++) {
        const CeedScalar x_i = (CeedScalar)i / (n_x * (p - 1)), y_j = (CeedScalar)j / (n_y * (p - 1));

        // Distorted mesh, so that the element matrices differ between elements
        x_array[i + j * (n_x * (p - 1) + 1) + 0 * num_dofs] = x_i + 0.05 * sin(M_PI * x_i) * sin(M_PI * y_j);
        x_array[i + j * (n_x * (p - 1) + 1) + 1 * num_dofs] = y_j + 0.03 * sin(2 * M_PI * x_i) * sin(M_PI * y_j);
      }
    }
    CeedVectorSetArray(x, CEED_MEM_HOST, CEED_COPY_VALUES, x_array);
  }
  CeedVectorCreate(ceed, num_dofs, &u);
  {
    CeedScalar *u_array;

    CeedVectorGetArrayWrite(u, CEED_MEM_HOST, &u_array);
    for (CeedInt i = 0; i < num_dofs; i++) u_array[i] = sin(1.0 + i);
    CeedVectorRestoreArray(u, &u_array);
  }
  CeedVectorCreate(ceed, num_dofs, &v);
  CeedVectorCreate(ceed, num_dofs, &v_elem_mats);
  CeedVectorCreate(ceed, num_qpts, &q_data);
  CeedVectorCreate(ceed, num_qpts, &q_data_copy);
  CeedVectorCreate(ceed, num_qpts, &u_q);
  {
    CeedScalar *u_q_array;

    CeedVectorGetArrayWrite(u_q, CEED_MEM_HOST, &u_q_array);
    for (CeedInt i = 0; i < num_qpts; i++) u_q_array[i] = cos(1.0 + i);
    CeedVectorRestoreArray(u_q, &u_q_array);
  }
  CeedVectorCreate(ceed, num_qpts, &v_q);
  CeedVectorCreate(ceed, num_qpts, &v_q_elem_mats);

  // Restrictions
  for (CeedInt i = 0; i < num_elem; i++) {
    CeedInt col, row, offset;

    col    = i % n_x;
    row    = i / n_x;
    offset = col * (p - 1) + row * (n_x * (p - 1) + 1) * (p - 1);
    for (CeedInt j = 0; j < p; j++) {
      for (CeedInt k = 0; k < p; k++) ind_u[p * (p * i + k) + j] = offset + k * (n_x * (p - 1) + 1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, num_elem, p * p, dim, num_dofs, dim * num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_x);
  CeedElemRestrictionCreate(ceed, num_elem, p * p, 1, 1, num_dofs, CEED_MEM_HOST, CEED_USE_POINTER, ind_u, &elem_restriction_u);

  CeedInt strides_q_data[3] = {1, q * q, q * q};
  CeedElemRestrictionCreateStrided(ceed, num_elem, q * q, 1, num_qpts, strides_q_data, &elem_restriction_q_data);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, dim, dim, p, q, CEED_GAUSS, &basis_x);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u);
  CeedBasisCreateTensorH1Lagrange(ceed, dim, 1, p, q, CEED_GAUSS, &basis_u_copy);

  // QFunction context
  CeedQFunctionContextCreate(ceed, &ctx_mass);
  CeedQFunctionContextSetData(ctx_mass, CEED_MEM_HOST, CEED_COPY_VALUES, sizeof(scale_context), &scale_context);
  CeedQFunctionContextRegisterDouble(ctx_mass, "scale", offsetof(ScaleContext, scale), 1, "scaling of the mass matrix");

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "dx", dim * dim, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, scaled_mass, scaled_mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionSetContext(qf_mass, ctx_mass);
  CeedQFunctionSetContextWritable(qf_mass, false);

  CeedQFunctionCreateInterior(ceed, 1, scaled_mass, scaled_mass_loc, &qf_mass_writable);
  CeedQFunctionAddInput(qf_mass_writable, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass_writable, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_writable, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionSetContext(qf_mass_writable, ctx_mass);

  CeedQFunctionCreateInterior(ceed, 1, mass_two_inputs, mass_two_inputs_loc, &qf_mass_two_inputs);
  CeedQFunctionAddInput(qf_mass_two_inputs, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass_two_inputs, "u 1", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddInput(qf_mass_two_inputs, "u 2", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_two_inputs, "v", 1, CEED_EVAL_INTERP);

  CeedQFunctionCreateInterior(ceed, 1, mass_passive_output, mass_passive_output_loc, &qf_mass_passive_output);
  CeedQFunctionAddInput(qf_mass_passive_output, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass_passive_output, "u", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_passive_output, "v", 1, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass_passive_output, "rho copy", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, scaled_mass, scaled_mass_loc, &qf_scale);
  CeedQFunctionAddInput(qf_scale, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_scale, "u", 1, CEED_EVAL_NONE);
  CeedQFunctionAddOutput(qf_scale, "v", 1, CEED_EVAL_NONE);
  CeedQFunctionSetContext(qf_scale, ctx_mass);
  CeedQFunctionSetContextWritable(qf_scale, false);

  // Operators
  CeedOperatorCreate(ceed, qf_setup, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_setup);
  CeedOperatorSetField(op_setup, "weight", CEED_ELEMRESTRICTION_NONE, basis_x, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup, "dx", elem_restriction_x, basis_x, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup, "rho", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);

  CeedOperator *ops_mass[4] = {&op_mass, &op_mass_elem_mats, &op_mass_2, &op_mass_elem_mats_2};
  for (CeedInt i = 0; i < 4; i++) {
    CeedOperatorCreate(ceed, qf_mass, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, ops_mass[i]);
    CeedOperatorSetField(*ops_mass[i], "rho", elem_restriction_q_data, CEED_BASIS_NONE, i < 2 ? q_data : q_data_copy);
    CeedOperatorSetField(*ops_mass[i], "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(*ops_mass[i], "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  }

  CeedOperatorCreate(ceed, qf_mass_writable, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_writable);
  CeedOperatorSetField(op_mass_writable, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_writable, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_writable, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass_two_inputs, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_two_inputs);
  CeedOperatorSetField(op_mass_two_inputs, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_two_inputs, "u 1", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_two_inputs, "u 2", elem_restriction_u, basis_u_copy, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_two_inputs, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);

  CeedOperatorCreate(ceed, qf_mass_passive_output, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, &op_mass_passive_output);
  CeedOperatorSetField(op_mass_passive_output, "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
  CeedOperatorSetField(op_mass_passive_output, "u", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_passive_output, "v", elem_restriction_u, basis_u, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_passive_output, "rho copy", elem_restriction_q_data, CEED_BASIS_NONE, q_data_copy);

  CeedOperator *ops_scale[2] = {&op_scale, &op_scale_elem_mats};
  for (CeedInt i = 0; i < 2; i++) {
    CeedOperatorCreate(ceed, qf_scale, CEED_QFUNCTION_NONE, CEED_QFUNCTION_NONE, ops_scale[i]);
    CeedOperatorSetField(*ops_scale[i], "rho", elem_restriction_q_data, CEED_BASIS_NONE, q_data);
    CeedOperatorSetField(*ops_scale[i], "u", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
    CeedOperatorSetField(*ops_scale[i], "v", elem_restriction_q_data, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE);
  }

  // Apply setup operator
  CeedOperatorApply(op_setup, x, q_data, CEED_REQUEST_IMMEDIATE);
  CeedVectorCopy(q_data, q_data_copy);

  // Apply with element matrices
  CeedOperatorSetLinearElementAssembly(op_mass_elem_mats, true);
  {
    bool use_linear_element_assembly;

    CeedOperatorGetLinearElementAssembly(op_mass_elem_mats, &use_linear_element_assembly);
    if (!use_linear_element_assembly) printf("Incorrect linear element assembly flag\n");
  }
  CheckApply(op_mass, op_mass_elem_mats, u, v, v_elem_mats, "Initial");

  // Update context
  CeedOperatorGetContextFieldLabel(op_mass_elem_mats, "scale", &scale_label);
  {
    double scale = 3.0;

    CeedOperatorSetContextDouble(op_mass_elem_mats, scale_label, &scale);
  }
  CheckApply(op_mass, op_mass_elem_mats, u, v, v_elem_mats, "Context update");

  // Update passive input
  CeedVectorScale(q_data, 0.5);
  CheckApply(op_mass, op_mass_elem_mats, u, v, v_elem_mats, "Passive input update");

  // Apply add
  {
    CeedSize          length;
    const CeedScalar *v_array, *v_elem_mats_array;

    CeedOperatorApply(op_mass, u, v, CEED_REQUEST_IMMEDIATE);
    CeedVectorSetValue(v_elem_mats, 1.0);
    CeedOperatorApplyAdd(op_mass_elem_mats, u, v_elem_mats, CEED_REQUEST_IMMEDIATE);
    CeedVectorGetLength(v, &length);
    CeedVectorGetArrayRead(v, CEED_MEM_HOST, &v_array);
    CeedVectorGetArrayRead(v_elem_mats, CEED_MEM_HOST, &v_elem_mats_array);
    for (CeedSize i = 0; i < length; i++) {
      if (fabs(v_array[i] + 1.0 - v_elem_mats_array[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Apply add: [%" CeedSize_FMT "] Error in application with element matrices: %f != %f\n", i, v_elem_mats_array[i], v_array[i] + 1.0);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(v, &v_array);
    CeedVectorRestoreArrayRead(v_elem_mats, &v_elem_mats_array);
  }

  // Diagonal assembly, which restricts the passive inputs again after the element matrices were assembled
  {
    CeedSize          length;
    CeedVector        diag, diag_elem_mats;
    const CeedScalar *diag_array, *diag_elem_mats_array;

    CeedVectorCreate(ceed, num_dofs, &diag);
    CeedVectorCreate(ceed, num_dofs, &diag_elem_mats);
    CeedOperatorLinearAssembleDiagonal(op_mass, diag, CEED_REQUEST_IMMEDIATE);
    CeedOperatorLinearAssembleDiagonal(op_mass_elem_mats, diag_elem_mats, CEED_REQUEST_IMMEDIATE);
    CeedVectorGetLength(diag, &length);
    CeedVectorGetArrayRead(diag, CEED_MEM_HOST, &diag_array);
    CeedVectorGetArrayRead(diag_elem_mats, CEED_MEM_HOST, &diag_elem_mats_array);
    for (CeedSize i = 0; i < length; i++) {
      if (fabs(diag_array[i] - diag_elem_mats_array[i]) > 100. * CEED_EPSILON) {
        // LCOV_EXCL_START
        printf("Diagonal: [%" CeedSize_FMT "] Error in assembly after application with element matrices: %f != %f\n", i, diag_elem_mats_array[i],
               diag_array[i]);
        // LCOV_EXCL_STOP
      }
    }
    CeedVectorRestoreArrayRead(diag, &diag_array);
    CeedVectorRestoreArrayRead(diag_elem_mats, &diag_elem_mats_array);
    CeedVectorDestroy(&diag);
    CeedVectorDestroy(&diag_elem_mats);
  }

  // Composite operator
  CeedOperatorCreateComposite(ceed, &op_composite);
  CeedOperatorCompositeAddSub(op_composite, op_mass);
  CeedOperatorCompositeAddSub(op_composite, op_mass_2);
  CeedOperatorCreateComposite(ceed, &op_composite_elem_mats);
  CeedOperatorCompositeAddSub(op_composite_elem_mats, op_mass_elem_mats);
  CeedOperatorCompositeAddSub(op_composite_elem_mats, op_mass_elem_mats_2);
  CeedOperatorSetLinearElementAssembly(op_composite_elem_mats, true);
  {
    bool use_linear_element_assembly;

    CeedOperatorGetLinearElementAssembly(op_mass_elem_mats_2, &use_linear_element_assembly);
    if (!use_linear_element_assembly) printf("Incorrect linear element assembly flag for sub-operator\n");
    CeedOperatorGetLinearElementAssembly(op_composite_elem_mats, &use_linear_element_assembly);
    if (!use_linear_element_assembly) printf("Incorrect linear element assembly flag for composite operator\n");
    CeedOperatorGetLinearElementAssembly(op_composite, &use_linear_element_assembly);
    if (use_linear_element_assembly) printf("Incorrect linear element assembly flag for composite operator without it\n");
  }
  CheckApply(op_composite, op_composite_elem_mats, u, v, v_elem_mats, "Composite");

  // Turn off
  CeedOperatorSetLinearElementAssembly(op_composite_elem_mats, false);
  {
    bool use_linear_element_assembly;

    CeedOperatorGetLinearElementAssembly(op_mass_elem_mats, &use_linear_element_assembly);
    if (use_linear_element_assembly) printf("Incorrect linear element assembly flag after turning off\n");
  }
  CheckApply(op_composite, op_composite_elem_mats, u, v, v_elem_mats, "Turned off");

  // Quadrature point operator, without bases
  CeedOperatorSetLinearElementAssembly(op_scale_elem_mats, true);
  CheckApply(op_scale, op_scale_elem_mats, u_q, v_q, v_q_elem_mats, "Quadrature point operator");

  // Unsupported operators
  {
    int         ierr;
    const char *err_msg;

    CeedSetErrorHandler(ceed, CeedErrorStore);
    ierr = CeedOperatorSetLinearElementAssembly(op_mass_writable, true);
    if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Writable context not detected\n");
    CeedResetErrorMessage(ceed, &err_msg);
    ierr = CeedOperatorSetLinearElementAssembly(op_mass_two_inputs, true);
    if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Multiple active bases not detected\n");
    CeedResetErrorMessage(ceed, &err_msg);
    ierr = CeedOperatorSetLinearElementAssembly(op_mass_passive_output, true);
    if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Passive output not detected\n");
    CeedResetErrorMessage(ceed, &err_msg);
    {
      // A composite operator with an unsupported sub-operator is left unchanged
      bool         use_linear_element_assembly;
      CeedOperator op_composite_unsupported;

      CeedOperatorCreateComposite(ceed, &op_composite_unsupported);
      CeedOperatorCompositeAddSub(op_composite_unsupported, op_mass_2);
      CeedOperatorCompositeAddSub(op_composite_unsupported, op_mass_writable);
      ierr = CeedOperatorSetLinearElementAssembly(op_composite_unsupported, true);
      if (ierr != CEED_ERROR_INCOMPATIBLE) printf("Unsupported sub-operator not detected\n");
      CeedResetErrorMessage(ceed, &err_msg);
      CeedOperatorGetLinearElementAssembly(op_mass_2, &use_linear_element_assembly);
      if (use_linear_element_assembly) printf("Sub-operator changed by a failed linear element assembly of its composite operator\n");
      CeedOperatorDestroy(&op_composite_unsupported);
    }
    CeedSetErrorHandler(ceed, CeedErrorAbort);
  }

  // Cleanup
  CeedVectorDestroy(&x);
  CeedVectorDestroy(&u);
  CeedVectorDestroy(&v);
  CeedVectorDestroy(&v_elem_mats);
  CeedVectorDestroy(&q_data);
  CeedVectorDestroy(&q_data_copy);
  CeedVectorDestroy(&u_q);
  CeedVectorDestroy(&v_q);
  CeedVectorDestroy(&v_q_elem_mats);
  CeedElemRestrictionDestroy(&elem_restriction_x);
  CeedElemRestrictionDestroy(&elem_restriction_u);
  CeedElemRestrictionDestroy(&elem_restriction_q_data);
  CeedBasisDestroy(&basis_x);
  CeedBasisDestroy(&basis_u);
  CeedBasisDestroy(&basis_u_copy);
  CeedQFunctionContextDestroy(&ctx_mass);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedQFunctionDestroy(&qf_mass_writable);
  CeedQFunctionDestroy(&qf_mass_two_inputs);
  CeedQFunctionDestroy(&qf_mass_passive_output);
  CeedQFunctionDestroy(&qf_scale);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_mass_elem_mats);
  CeedOperatorDestroy(&op_mass_2);
  CeedOperatorDestroy(&op_mass_elem_mats_2);
  CeedOperatorDestroy(&op_composite);
  CeedOperatorDestroy(&op_composite_elem_mats);
  CeedOperatorDestroy(&op_mass_writable);
  CeedOperatorDestroy(&op_mass_two_inputs);
  CeedOperatorDestroy(&op_mass_passive_output);
  CeedOperatorDestroy(&op_scale);
  CeedOperatorDestroy(&op_scale_elem_mats);
  CeedDestroy(&ceed);
  return 0;
}
