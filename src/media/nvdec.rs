//! NVIDIA Video Decoder capability queries.
//!
//! This module binds the minimal NVDECODE surface needed before creating full
//! decoder sessions: device-scoped capability checks through `libnvcuvid`.

use std::ffi::{c_int, c_uchar, c_uint, c_ushort};

use candle::{Device, Result, cuda::WrapErr};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecCodec {
    H264,
    Hevc,
    Av1,
    Vp9,
}

impl NvDecCodec {
    fn raw(self) -> u32 {
        match self {
            Self::H264 => 4,
            Self::Hevc => 8,
            Self::Vp9 => 10,
            Self::Av1 => 11,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::H264 => "H.264",
            Self::Hevc => "HEVC",
            Self::Av1 => "AV1",
            Self::Vp9 => "VP9",
        }
    }
}

impl TryFrom<u32> for NvDecCodec {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::H264),
            1 => Ok(Self::Hevc),
            2 => Ok(Self::Av1),
            3 => Ok(Self::Vp9),
            other => candle::bail!(
                "unknown NVDECODE codec {other}; expected 0=H264, 1=HEVC, 2=AV1, or 3=VP9"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvDecCaps {
    pub codec: NvDecCodec,
    pub chroma_format: NvDecChromaFormat,
    pub bit_depth_minus_8: u32,
    pub supported: bool,
    pub nvdec_count: usize,
    pub output_format_mask: u32,
    pub min_width: usize,
    pub min_height: usize,
    pub max_width: usize,
    pub max_height: usize,
    pub max_macroblock_count: usize,
    pub histogram_supported: bool,
    pub histogram_counter_bit_depth: usize,
    pub max_histogram_bins: usize,
}

impl NvDecCaps {
    pub fn supports(self, format: NvDecSurfaceFormat) -> bool {
        self.output_format_mask & format.mask_bit() != 0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecChromaFormat {
    Monochrome,
    Yuv420,
    Yuv422,
    Yuv444,
}

impl NvDecChromaFormat {
    fn raw(self) -> u32 {
        match self {
            Self::Monochrome => 0,
            Self::Yuv420 => 1,
            Self::Yuv422 => 2,
            Self::Yuv444 => 3,
        }
    }
}

impl TryFrom<u32> for NvDecChromaFormat {
    type Error = candle::Error;

    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Monochrome),
            1 => Ok(Self::Yuv420),
            2 => Ok(Self::Yuv422),
            3 => Ok(Self::Yuv444),
            other => candle::bail!(
                "unknown NVDECODE chroma format {other}; expected 0=monochrome, 1=YUV420, 2=YUV422, or 3=YUV444"
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NvDecSurfaceFormat {
    Nv12,
    P016,
    Yuv444,
    Yuv44416Bit,
}

impl NvDecSurfaceFormat {
    fn raw(self) -> u32 {
        match self {
            Self::Nv12 => 0,
            Self::P016 => 1,
            Self::Yuv444 => 2,
            Self::Yuv44416Bit => 3,
        }
    }

    fn mask_bit(self) -> u32 {
        1 << self.raw()
    }
}

pub fn query_caps_420(
    device: &Device,
    codec: NvDecCodec,
    bit_depth_minus_8: u32,
) -> Result<NvDecCaps> {
    query_caps(device, codec, NvDecChromaFormat::Yuv420, bit_depth_minus_8)
}

pub fn query_caps(
    device: &Device,
    codec: NvDecCodec,
    chroma_format: NvDecChromaFormat,
    bit_depth_minus_8: u32,
) -> Result<NvDecCaps> {
    if !device.is_cuda() {
        candle::bail!("NVDECODE capability query requires a CUDA Candle device");
    }
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    stream.context().bind_to_thread().w()?;

    let mut raw = CuvidDecodeCaps {
        e_codec_type: codec.raw(),
        e_chroma_format: chroma_format.raw(),
        n_bit_depth_minus_8: bit_depth_minus_8,
        ..Default::default()
    };
    check_cuvid(
        unsafe { cuvidGetDecoderCaps(&mut raw as *mut CuvidDecodeCaps) },
        "cuvidGetDecoderCaps",
    )?;

    Ok(NvDecCaps {
        codec,
        chroma_format,
        bit_depth_minus_8,
        supported: raw.b_is_supported != 0,
        nvdec_count: raw.n_num_nvdecs as usize,
        output_format_mask: raw.n_output_format_mask as u32,
        min_width: raw.n_min_width as usize,
        min_height: raw.n_min_height as usize,
        max_width: raw.n_max_width as usize,
        max_height: raw.n_max_height as usize,
        max_macroblock_count: raw.n_max_mb_count as usize,
        histogram_supported: raw.b_is_histogram_supported != 0,
        histogram_counter_bit_depth: raw.n_counter_bit_depth as usize,
        max_histogram_bins: raw.n_max_histogram_bins as usize,
    })
}

fn check_cuvid(status: c_int, context: &str) -> Result<()> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        candle::bail!("{context} failed with CUDA driver status {status}")
    }
}

const CUDA_SUCCESS: c_int = 0;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CuvidDecodeCaps {
    e_codec_type: c_uint,
    e_chroma_format: c_uint,
    n_bit_depth_minus_8: c_uint,
    reserved1: [c_uint; 3],
    b_is_supported: c_uchar,
    n_num_nvdecs: c_uchar,
    n_output_format_mask: c_ushort,
    n_max_width: c_uint,
    n_max_height: c_uint,
    n_max_mb_count: c_uint,
    n_min_width: c_ushort,
    n_min_height: c_ushort,
    b_is_histogram_supported: c_uchar,
    n_counter_bit_depth: c_uchar,
    n_max_histogram_bins: c_ushort,
    reserved3: [c_uint; 10],
}

impl Default for CuvidDecodeCaps {
    fn default() -> Self {
        Self {
            e_codec_type: 0,
            e_chroma_format: 0,
            n_bit_depth_minus_8: 0,
            reserved1: [0; 3],
            b_is_supported: 0,
            n_num_nvdecs: 0,
            n_output_format_mask: 0,
            n_max_width: 0,
            n_max_height: 0,
            n_max_mb_count: 0,
            n_min_width: 0,
            n_min_height: 0,
            b_is_histogram_supported: 0,
            n_counter_bit_depth: 0,
            n_max_histogram_bins: 0,
            reserved3: [0; 10],
        }
    }
}

#[link(name = "nvcuvid")]
unsafe extern "C" {
    fn cuvidGetDecoderCaps(caps: *mut CuvidDecodeCaps) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queries_h264_420_8bit_caps_on_cuda() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let caps = query_caps_420(&device, NvDecCodec::H264, 0)?;

        println!("NVDECODE {} 8-bit 4:2:0 caps: {caps:?}", caps.codec.name());
        assert!(caps.supported);
        assert!(caps.nvdec_count > 0);
        assert!(caps.max_width >= caps.min_width);
        assert!(caps.max_height >= caps.min_height);
        assert!(caps.supports(NvDecSurfaceFormat::Nv12));
        Ok(())
    }

    #[test]
    fn unsupported_codec_ids_are_rejected() {
        let err = NvDecCodec::try_from(99).unwrap_err();
        assert!(err.to_string().contains("unknown NVDECODE codec"));
    }
}
