#ifndef STABLE_WORLDMODEL_CANDLE_H
#define STABLE_WORLDMODEL_CANDLE_H

#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct SwmTdMpc2 SwmTdMpc2;
typedef struct SwmLeWm SwmLeWm;
typedef struct SwmCudaImage SwmCudaImage;
typedef struct SwmCudaNv12 SwmCudaNv12;
typedef struct SwmNvDecDecoder SwmNvDecDecoder;

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

typedef enum SwmPackedImageFormat {
  SWM_PACKED_IMAGE_FORMAT_RGB = 0,
  SWM_PACKED_IMAGE_FORMAT_BGR = 1,
  SWM_PACKED_IMAGE_FORMAT_RGBA = 2,
  SWM_PACKED_IMAGE_FORMAT_BGRA = 3,
} SwmPackedImageFormat;

typedef enum SwmNv12ColorSpace {
  SWM_NV12_COLOR_SPACE_BT601_VIDEO = 0,
  SWM_NV12_COLOR_SPACE_BT709_VIDEO = 1,
  SWM_NV12_COLOR_SPACE_BT601_FULL = 2,
  SWM_NV12_COLOR_SPACE_BT709_FULL = 3,
} SwmNv12ColorSpace;

typedef enum SwmNvDecCodec {
  SWM_NVDEC_CODEC_H264 = 0,
  SWM_NVDEC_CODEC_HEVC = 1,
  SWM_NVDEC_CODEC_AV1 = 2,
  SWM_NVDEC_CODEC_VP9 = 3,
} SwmNvDecCodec;

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

typedef struct SwmNvDecCaps {
  int supported;
  size_t nvdec_count;
  size_t min_width;
  size_t min_height;
  size_t max_width;
  size_t max_height;
  size_t max_macroblock_count;
  unsigned int output_format_mask;
  int supports_nv12;
  int supports_p016;
  int supports_yuv444;
  int supports_yuv444_16bit;
  int histogram_supported;
  size_t histogram_counter_bit_depth;
  size_t max_histogram_bins;
} SwmNvDecCaps;

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

SwmStatus swm_cuda_image_alloc(const char *device,
                               size_t batch,
                               size_t height,
                               size_t width,
                               SwmPackedImageFormat format,
                               SwmCudaImage **out);

void swm_cuda_image_free(SwmCudaImage *handle);

SwmStatus swm_cuda_image_ptr(const SwmCudaImage *handle,
                             void **out,
                             size_t *pitch_bytes_out);

SwmStatus swm_cuda_nv12_alloc(const char *device,
                              size_t batch,
                              size_t height,
                              size_t width,
                              SwmCudaNv12 **out);

void swm_cuda_nv12_free(SwmCudaNv12 *handle);

SwmStatus swm_cuda_nv12_y_ptr(const SwmCudaNv12 *handle,
                              void **out,
                              size_t *pitch_bytes_out);

SwmStatus swm_cuda_nv12_uv_ptr(const SwmCudaNv12 *handle,
                               void **out,
                               size_t *pitch_bytes_out);

SwmStatus swm_nvdec_query_420(const char *device,
                              SwmNvDecCodec codec,
                              unsigned int bit_depth_minus_8,
                              SwmNvDecCaps *out);

SwmStatus swm_nvdec_decoder_create_420(const char *device,
                                       SwmNvDecCodec codec,
                                       size_t width,
                                       size_t height,
                                       size_t decode_surfaces,
                                       size_t output_surfaces,
                                       SwmNvDecDecoder **out);

void swm_nvdec_decoder_free(SwmNvDecDecoder *handle);

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

SwmStatus swm_tdmpc2_reset_cuda_image(SwmTdMpc2 *handle,
                                      const SwmCudaImage *image);

SwmStatus swm_tdmpc2_reset_state_cuda_image(SwmTdMpc2 *handle,
                                            const float *state,
                                            size_t batch,
                                            size_t state_dim,
                                            const SwmCudaImage *image);

SwmStatus swm_tdmpc2_reset_cuda_nv12(SwmTdMpc2 *handle,
                                     const SwmCudaNv12 *nv12,
                                     SwmNv12ColorSpace color_space);

SwmStatus swm_tdmpc2_reset_state_cuda_nv12(SwmTdMpc2 *handle,
                                           const float *state,
                                           size_t batch,
                                           size_t state_dim,
                                           const SwmCudaNv12 *nv12,
                                           SwmNv12ColorSpace color_space);

SwmStatus swm_lewm_reset_cuda_image_history(SwmLeWm *handle,
                                            const SwmCudaImage *image,
                                            size_t batch,
                                            size_t time);

SwmStatus swm_lewm_set_goal_cuda_image_history(SwmLeWm *handle,
                                               const SwmCudaImage *image,
                                               size_t batch,
                                               size_t time);

SwmStatus swm_lewm_reset_cuda_nv12_history(SwmLeWm *handle,
                                           const SwmCudaNv12 *nv12,
                                           size_t batch,
                                           size_t time,
                                           SwmNv12ColorSpace color_space);

SwmStatus swm_lewm_set_goal_cuda_nv12_history(SwmLeWm *handle,
                                              const SwmCudaNv12 *nv12,
                                              size_t batch,
                                              size_t time,
                                              SwmNv12ColorSpace color_space);

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
