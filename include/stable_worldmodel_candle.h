#ifndef STABLE_WORLDMODEL_CANDLE_H
#define STABLE_WORLDMODEL_CANDLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SwmTdMpc2 SwmTdMpc2;

typedef enum SwmStatus {
  SWM_STATUS_OK = 0,
  SWM_STATUS_NULL_POINTER = 1,
  SWM_STATUS_INVALID_ARGUMENT = 2,
  SWM_STATUS_RUNTIME_ERROR = 3,
  SWM_STATUS_PANIC = 4,
} SwmStatus;

typedef struct SwmCemPlanConfig {
  size_t horizon;
  size_t samples;
  size_t elites;
  size_t iterations;
  float init_std;
  float min_std;
} SwmCemPlanConfig;

typedef struct SwmMppiPlanConfig {
  size_t horizon;
  size_t samples;
  size_t iterations;
  float noise_std;
  float temperature;
} SwmMppiPlanConfig;

const char *swm_last_error_message(void);

SwmStatus swm_tdmpc2_load(const char *artifact_dir,
                          const char *device,
                          const char *dtype,
                          SwmTdMpc2 **out);

void swm_tdmpc2_free(SwmTdMpc2 *handle);

SwmStatus swm_tdmpc2_state_dim(const SwmTdMpc2 *handle, size_t *out);

SwmStatus swm_tdmpc2_action_dim(const SwmTdMpc2 *handle, size_t *out);

SwmStatus swm_tdmpc2_reset_state(SwmTdMpc2 *handle,
                                 const float *state,
                                 size_t batch,
                                 size_t state_dim);

SwmStatus swm_tdmpc2_plan_cem(SwmTdMpc2 *handle,
                              SwmCemPlanConfig config,
                              float *action_out,
                              float *sequence_out,
                              float *best_cost_out);

SwmStatus swm_tdmpc2_plan_mppi(SwmTdMpc2 *handle,
                               SwmMppiPlanConfig config,
                               float *action_out,
                               float *sequence_out,
                               float *best_cost_out);

#ifdef __cplusplus
}
#endif

#endif
