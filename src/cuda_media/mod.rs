//! NVIDIA/CUDA media ingestion primitives.
//!
//! This module is intentionally CUDA-native: it owns reusable Candle tensors and
//! launches preprocessing kernels on the same CUDA stream Candle uses for model
//! execution.

use std::fmt;

use candle::{
    CudaStorage, DType, Device, InplaceOp2, Layout, Result, Storage, Tensor,
    backend::BackendStorage,
    cuda::{
        WrapErr,
        cudarc::{
            driver::{LaunchConfig, PushKernelArg},
            nvrtc,
        },
    },
    op::BackpropOp,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedImageShape {
    pub batch: usize,
    pub height: usize,
    pub width: usize,
    pub format: PackedImageFormat,
}

impl PackedImageShape {
    pub fn new(batch: usize, height: usize, width: usize, format: PackedImageFormat) -> Self {
        Self {
            batch,
            height,
            width,
            format,
        }
    }

    pub fn channels(self) -> usize {
        self.format.channels()
    }

    pub fn elem_count(self) -> usize {
        self.batch * self.height * self.width * self.channels()
    }

    fn validate(self) -> Result<()> {
        if self.batch == 0 || self.height == 0 || self.width == 0 {
            candle::bail!("packed CUDA image dimensions must be greater than zero");
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedImageFormat {
    Rgb,
    Bgr,
    Rgba,
    Bgra,
}

impl PackedImageFormat {
    pub fn channels(self) -> usize {
        match self {
            Self::Rgb | Self::Bgr => 3,
            Self::Rgba | Self::Bgra => 4,
        }
    }

    fn kernel_id(self) -> u32 {
        match self {
            Self::Rgb => 0,
            Self::Bgr => 1,
            Self::Rgba => 2,
            Self::Bgra => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CudaImagePreprocess {
    pub output_height: usize,
    pub output_width: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl CudaImagePreprocess {
    pub fn imagenet_224() -> Self {
        Self {
            output_height: 224,
            output_width: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    fn validate(self) -> Result<()> {
        if self.output_height == 0 || self.output_width == 0 {
            candle::bail!("CUDA preprocess output dimensions must be greater than zero");
        }
        if self.std.iter().any(|value| *value == 0.0) {
            candle::bail!("CUDA preprocess std values must be non-zero");
        }
        Ok(())
    }
}

pub struct CudaImagePreprocessor {
    input_shape: PackedImageShape,
    config: CudaImagePreprocess,
    output: Tensor,
}

impl fmt::Debug for CudaImagePreprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaImagePreprocessor")
            .field("input_shape", &self.input_shape)
            .field("config", &self.config)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl CudaImagePreprocessor {
    pub fn new(
        device: &Device,
        input_shape: PackedImageShape,
        config: CudaImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if !device.is_cuda() {
            candle::bail!("CudaImagePreprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            config,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn preprocess_packed_u8(&mut self, input: &Tensor) -> Result<&Tensor> {
        validate_input_tensor(input, self.input_shape)?;
        let op = PackedU8ToNchwF32 {
            input_shape: self.input_shape,
            config: self.config,
            output_layout: CudaMediaOutputLayout::Latest,
        };
        self.output.inplace_op2(input, &op)?;
        Ok(&self.output)
    }
}

pub struct CudaImageHistoryPreprocessor {
    input_shape: PackedImageShape,
    config: CudaImagePreprocess,
    history_len: usize,
    output: Tensor,
}

impl fmt::Debug for CudaImageHistoryPreprocessor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CudaImageHistoryPreprocessor")
            .field("input_shape", &self.input_shape)
            .field("config", &self.config)
            .field("history_len", &self.history_len)
            .field("output_shape", &self.output.shape())
            .finish()
    }
}

impl CudaImageHistoryPreprocessor {
    pub fn new(
        device: &Device,
        input_shape: PackedImageShape,
        history_len: usize,
        config: CudaImagePreprocess,
    ) -> Result<Self> {
        input_shape.validate()?;
        config.validate()?;
        if history_len == 0 {
            candle::bail!("CUDA image history length must be greater than zero");
        }
        if !device.is_cuda() {
            candle::bail!("CudaImageHistoryPreprocessor requires a CUDA Candle device");
        }

        let output = Tensor::zeros(
            (
                input_shape.batch,
                history_len,
                3,
                config.output_height,
                config.output_width,
            ),
            DType::F32,
            device,
        )?;

        Ok(Self {
            input_shape,
            config,
            history_len,
            output,
        })
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn preprocess_packed_u8_into_slot(
        &mut self,
        input: &Tensor,
        history_slot: usize,
    ) -> Result<&Tensor> {
        validate_input_tensor(input, self.input_shape)?;
        if history_slot >= self.history_len {
            candle::bail!(
                "CUDA image history slot {history_slot} is outside history_len {}",
                self.history_len
            );
        }
        let op = PackedU8ToNchwF32 {
            input_shape: self.input_shape,
            config: self.config,
            output_layout: CudaMediaOutputLayout::History {
                history_len: self.history_len,
                history_slot,
            },
        };
        self.output.inplace_op2(input, &op)?;
        Ok(&self.output)
    }
}

fn validate_input_tensor(input: &Tensor, shape: PackedImageShape) -> Result<()> {
    if !input.device().is_cuda() {
        candle::bail!("packed image input must live on a CUDA Candle device");
    }
    if input.dtype() != DType::U8 {
        candle::bail!(
            "packed image input must use U8 dtype, got {:?}",
            input.dtype()
        );
    }
    let expected = [shape.batch, shape.height, shape.width, shape.channels()];
    if input.dims() != expected {
        candle::bail!(
            "packed image input shape mismatch: expected {:?}, got {:?}",
            expected,
            input.dims()
        );
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct PackedU8ToNchwF32 {
    input_shape: PackedImageShape,
    config: CudaImagePreprocess,
    output_layout: CudaMediaOutputLayout,
}

#[derive(Debug, Clone, Copy)]
enum CudaMediaOutputLayout {
    Latest,
    History {
        history_len: usize,
        history_slot: usize,
    },
}

impl InplaceOp2 for PackedU8ToNchwF32 {
    fn name(&self) -> &'static str {
        "packed-u8-to-nchw-f32"
    }

    fn cpu_fwd(
        &self,
        _output: &mut candle::CpuStorage,
        _output_layout: &Layout,
        _input: &candle::CpuStorage,
        _input_layout: &Layout,
    ) -> Result<()> {
        candle::bail!("packed-u8-to-nchw-f32 is a CUDA media operation")
    }

    fn cuda_fwd(
        &self,
        output: &mut CudaStorage,
        output_layout: &Layout,
        input: &CudaStorage,
        input_layout: &Layout,
    ) -> Result<()> {
        validate_output_layout(
            output,
            output_layout,
            self.input_shape,
            self.config,
            self.output_layout,
        )?;
        validate_input_layout(input, input_layout, self.input_shape)?;

        let cuda = output.device.clone();
        let input = input.as_cuda_slice::<u8>()?;
        let mut output = output.as_cuda_slice_mut::<f32>()?;
        let input = contiguous_slice(input, input_layout, "packed image input")?;
        let mut output = contiguous_slice_mut(&mut output, output_layout, "model image output")?;

        let ptx = nvrtc::safe::compile_ptx_with_opts(
            PACKED_U8_TO_NCHW_F32_CUDA,
            nvrtc::CompileOptions {
                use_fast_math: Some(true),
                ..Default::default()
            },
        )
        .w()?;
        let func = cuda.get_or_load_custom_func(
            "swm_packed_u8_to_nchw_f32",
            "swm_cuda_media_preprocess",
            &ptx.to_src(),
        )?;

        let elem_count =
            self.input_shape.batch * 3 * self.config.output_height * self.config.output_width;
        let cfg = LaunchConfig::for_num_elems(elem_count as u32);
        let elem_count_u32 = elem_count as u32;
        let batch = self.input_shape.batch as u32;
        let in_h = self.input_shape.height as u32;
        let in_w = self.input_shape.width as u32;
        let channels = self.input_shape.channels() as u32;
        let out_h = self.config.output_height as u32;
        let out_w = self.config.output_width as u32;
        let format = self.input_shape.format.kernel_id();
        let (history_len, history_slot) = match self.output_layout {
            CudaMediaOutputLayout::Latest => (0u32, 0u32),
            CudaMediaOutputLayout::History {
                history_len,
                history_slot,
            } => (history_len as u32, history_slot as u32),
        };
        let mut builder = func.builder();
        builder.arg(&input);
        builder.arg(&mut output);
        builder.arg(&elem_count_u32);
        builder.arg(&batch);
        builder.arg(&in_h);
        builder.arg(&in_w);
        builder.arg(&channels);
        builder.arg(&out_h);
        builder.arg(&out_w);
        builder.arg(&format);
        builder.arg(&history_len);
        builder.arg(&history_slot);
        builder.arg(&self.config.mean[0]);
        builder.arg(&self.config.mean[1]);
        builder.arg(&self.config.mean[2]);
        builder.arg(&self.config.std[0]);
        builder.arg(&self.config.std[1]);
        builder.arg(&self.config.std[2]);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(())
    }
}

fn validate_output_layout(
    output: &CudaStorage,
    layout: &Layout,
    input_shape: PackedImageShape,
    config: CudaImagePreprocess,
    output_layout: CudaMediaOutputLayout,
) -> Result<()> {
    if output.dtype() != DType::F32 {
        candle::bail!("CUDA media output tensor must use F32 dtype");
    }
    let expected = match output_layout {
        CudaMediaOutputLayout::Latest => vec![
            input_shape.batch,
            3,
            config.output_height,
            config.output_width,
        ],
        CudaMediaOutputLayout::History { history_len, .. } => vec![
            input_shape.batch,
            history_len,
            3,
            config.output_height,
            config.output_width,
        ],
    };
    if layout.dims() != expected {
        candle::bail!(
            "CUDA media output shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA media output")?;
    Ok(())
}

fn validate_input_layout(
    input: &CudaStorage,
    layout: &Layout,
    input_shape: PackedImageShape,
) -> Result<()> {
    if input.dtype() != DType::U8 {
        candle::bail!("CUDA media input tensor must use U8 dtype");
    }
    let expected = [
        input_shape.batch,
        input_shape.height,
        input_shape.width,
        input_shape.channels(),
    ];
    if layout.dims() != expected {
        candle::bail!(
            "CUDA media input shape mismatch: expected {:?}, got {:?}",
            expected,
            layout.dims()
        );
    }
    require_contiguous(layout, "CUDA media input")?;
    Ok(())
}

fn require_contiguous(layout: &Layout, name: &str) -> Result<()> {
    if layout.contiguous_offsets().is_none() {
        candle::bail!("{name} must be contiguous");
    }
    Ok(())
}

fn contiguous_slice<'a, T>(
    slice: &'a candle::cuda::cudarc::driver::CudaSlice<T>,
    layout: &Layout,
    name: &str,
) -> Result<candle::cuda::cudarc::driver::CudaView<'a, T>> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("{name} must be contiguous");
    };
    Ok(slice.slice(start..end))
}

fn contiguous_slice_mut<'a, T>(
    slice: &'a mut candle::cuda::cudarc::driver::CudaSlice<T>,
    layout: &Layout,
    name: &str,
) -> Result<candle::cuda::cudarc::driver::CudaViewMut<'a, T>> {
    let Some((start, end)) = layout.contiguous_offsets() else {
        candle::bail!("{name} must be contiguous");
    };
    Ok(slice.slice_mut(start..end))
}

pub fn packed_u8_tensor_from_host(
    bytes: &[u8],
    shape: PackedImageShape,
    device: &Device,
) -> Result<Tensor> {
    shape.validate()?;
    if !device.is_cuda() {
        candle::bail!("packed_u8_tensor_from_host requires a CUDA Candle device");
    }
    if bytes.len() != shape.elem_count() {
        candle::bail!(
            "packed image buffer has {} bytes, expected {}",
            bytes.len(),
            shape.elem_count()
        );
    }
    Tensor::from_slice(
        bytes,
        (shape.batch, shape.height, shape.width, shape.channels()),
        device,
    )
}

pub fn tensor_from_cuda_slice_f32(
    slice: candle::cuda::cudarc::driver::CudaSlice<f32>,
    shape: impl Into<candle::Shape>,
    device: candle::CudaDevice,
) -> Tensor {
    let storage = CudaStorage::wrap_cuda_slice(slice, device);
    Tensor::from_storage(Storage::Cuda(storage), shape, BackpropOp::none(), false)
}

const PACKED_U8_TO_NCHW_F32_CUDA: &str = r#"
extern "C" __global__ void swm_packed_u8_to_nchw_f32(
    const unsigned char* __restrict__ input,
    float* __restrict__ output,
    unsigned int elem_count,
    unsigned int batch,
    unsigned int in_h,
    unsigned int in_w,
    unsigned int channels,
    unsigned int out_h,
    unsigned int out_w,
    unsigned int format,
    unsigned int history_len,
    unsigned int history_slot,
    float mean0,
    float mean1,
    float mean2,
    float std0,
    float std1,
    float std2
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elem_count) {
        return;
    }

    unsigned int x = idx % out_w;
    unsigned int tmp = idx / out_w;
    unsigned int y = tmp % out_h;
    tmp /= out_h;
    unsigned int c = tmp % 3;
    unsigned int b = tmp / 3;
    if (b >= batch) {
        return;
    }

    float src_y = ((float)y + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;
    float src_x = ((float)x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
    if (src_y < 0.0f) {
        src_y = 0.0f;
    }
    if (src_x < 0.0f) {
        src_x = 0.0f;
    }

    unsigned int y0 = (unsigned int)floorf(src_y);
    unsigned int x0 = (unsigned int)floorf(src_x);
    unsigned int y1 = y0 + 1 < in_h ? y0 + 1 : y0;
    unsigned int x1 = x0 + 1 < in_w ? x0 + 1 : x0;
    float wy = src_y - (float)y0;
    float wx = src_x - (float)x0;

    unsigned int src_c = c;
    if (format == 1u || format == 3u) {
        src_c = 2u - c;
    }

    unsigned int base00 = (((b * in_h + y0) * in_w + x0) * channels + src_c);
    unsigned int base01 = (((b * in_h + y0) * in_w + x1) * channels + src_c);
    unsigned int base10 = (((b * in_h + y1) * in_w + x0) * channels + src_c);
    unsigned int base11 = (((b * in_h + y1) * in_w + x1) * channels + src_c);

    float v00 = (float)input[base00];
    float v01 = (float)input[base01];
    float v10 = (float)input[base10];
    float v11 = (float)input[base11];
    float top = v00 + (v01 - v00) * wx;
    float bottom = v10 + (v11 - v10) * wx;
    float value = (top + (bottom - top) * wy) * 0.00392156862745098f;

    float mean = c == 0u ? mean0 : (c == 1u ? mean1 : mean2);
    float std = c == 0u ? std0 : (c == 1u ? std1 : std2);
    float normalized = (value - mean) / std;
    if (history_len == 0u) {
        output[idx] = normalized;
    } else {
        unsigned int out_idx =
            ((((b * history_len + history_slot) * 3u + c) * out_h + y) * out_w + x);
        output[out_idx] = normalized;
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocesses_packed_rgb_to_nchw_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = PackedImageShape::new(1, 2, 2, PackedImageFormat::Rgb);
        let input = packed_u8_tensor_from_host(
            &[
                255, 0, 0, //
                0, 255, 0, //
                0, 0, 255, //
                255, 255, 255,
            ],
            shape,
            &device,
        )?;
        let config = CudaImagePreprocess {
            output_height: 2,
            output_width: 2,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor = CudaImagePreprocessor::new(&device, shape, config)?;
        let output = preprocessor.preprocess_packed_u8(&input)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = [
            1.0, 0.0, 0.0, 1.0, //
            0.0, 1.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, 1.0,
        ];

        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1e-6,
                "output[{idx}] expected {expected}, got {actual}, diff {diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn preprocesses_packed_rgb_into_history_slot_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let shape = PackedImageShape::new(1, 1, 1, PackedImageFormat::Bgr);
        let input = packed_u8_tensor_from_host(&[0, 128, 255], shape, &device)?;
        let config = CudaImagePreprocess {
            output_height: 1,
            output_width: 1,
            mean: [0.0, 0.0, 0.0],
            std: [1.0, 1.0, 1.0],
        };
        let mut preprocessor = CudaImageHistoryPreprocessor::new(&device, shape, 3, config)?;
        let output = preprocessor.preprocess_packed_u8_into_slot(&input, 1)?;
        let actual = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = [0.0, 0.0, 0.0, 1.0, 128.0 / 255.0, 0.0, 0.0, 0.0, 0.0];

        for (idx, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 1e-6,
                "output[{idx}] expected {expected}, got {actual}, diff {diff}"
            );
        }
        Ok(())
    }
}
