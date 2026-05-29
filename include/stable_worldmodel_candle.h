#ifndef STABLE_WORLDMODEL_CANDLE_H
#define STABLE_WORLDMODEL_CANDLE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SwmTdMpc2 SwmTdMpc2;
typedef struct SwmLeWm SwmLeWm;

typedef enum SwmStatus {
  SWM_STATUS_OK = 0,
  SWM_STATUS_NULL_POINTER = 1,
  SWM_STATUS_INVALID_ARGUMENT = 2,
  SWM_STATUS_RUNTIME_ERROR = 3,
  SWM_STATUS_PANIC = 4,
} SwmStatus;

typedef enum SwmPixelLayout {
  SWM_PIXEL_LAYOUT_NCHW = 0,
  SWM_PIXEL_LAYOUT_NHWC = 1,
} SwmPixelLayout;

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

typedef struct SwmIcemPlanConfig {
  size_t horizon;
  size_t samples;
  size_t elites;
  size_t keep_elites;
  size_t iterations;
  float init_std;
  float min_std;
} SwmIcemPlanConfig;

const char *swm_last_error_message(void);

SwmStatus swm_tdmpc2_load(const char *artifact_dir,
                          const char *device,
                          const char *dtype,
                          SwmTdMpc2 **out);

SwmStatus swm_lewm_load(const char *artifact_dir,
                        const char *device,
                        const char *dtype,
                        SwmLeWm **out);

void swm_tdmpc2_free(SwmTdMpc2 *handle);

void swm_lewm_free(SwmLeWm *handle);

SwmStatus swm_tdmpc2_state_dim(const SwmTdMpc2 *handle, size_t *out);

SwmStatus swm_tdmpc2_image_size(const SwmTdMpc2 *handle, size_t *out);

SwmStatus swm_tdmpc2_action_dim(const SwmTdMpc2 *handle, size_t *out);

SwmStatus swm_lewm_action_dim(const SwmLeWm *handle, size_t *out);

SwmStatus swm_lewm_image_size(const SwmLeWm *handle, size_t *out);

SwmStatus swm_lewm_history_size(const SwmLeWm *handle, size_t *out);

SwmStatus swm_tdmpc2_reset_state(SwmTdMpc2 *handle,
                                 const float *state,
                                 size_t batch,
                                 size_t state_dim);

SwmStatus swm_lewm_reset_pixels(SwmLeWm *handle,
                                const float *pixels,
                                size_t batch,
                                size_t time,
                                size_t height,
                                size_t width);

SwmStatus swm_lewm_set_goal_pixels(SwmLeWm *handle,
                                   const float *pixels,
                                   size_t batch,
                                   size_t time,
                                   size_t height,
                                   size_t width);

SwmStatus swm_tdmpc2_reset_pixels(SwmTdMpc2 *handle,
                                  const float *pixels,
                                  size_t batch,
                                  size_t height,
                                  size_t width,
                                  SwmPixelLayout layout);

SwmStatus swm_tdmpc2_reset_state_pixels(SwmTdMpc2 *handle,
                                        const float *state,
                                        const float *pixels,
                                        size_t batch,
                                        size_t state_dim,
                                        size_t height,
                                        size_t width,
                                        SwmPixelLayout layout);

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

SwmStatus swm_tdmpc2_plan_icem(SwmTdMpc2 *handle,
                               SwmIcemPlanConfig config,
                               float *action_out,
                               float *sequence_out,
                               float *best_cost_out);

SwmStatus swm_lewm_plan_cem(SwmLeWm *handle,
                            SwmCemPlanConfig config,
                            float *action_out,
                            float *sequence_out,
                            float *best_cost_out);

SwmStatus swm_lewm_plan_mppi(SwmLeWm *handle,
                             SwmMppiPlanConfig config,
                             float *action_out,
                             float *sequence_out,
                             float *best_cost_out);

SwmStatus swm_lewm_plan_icem(SwmLeWm *handle,
                             SwmIcemPlanConfig config,
                             float *action_out,
                             float *sequence_out,
                             float *best_cost_out);

SwmStatus swm_tdmpc2_clear_icem_warm_start(SwmTdMpc2 *handle);

SwmStatus swm_lewm_clear_icem_warm_start(SwmLeWm *handle);

#ifdef __cplusplus
}
#endif

#endif
