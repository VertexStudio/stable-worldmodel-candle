use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char, c_int},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
};

use candle::{DType, Device, Tensor};

use crate::{
    artifact::{DeploymentArtifact, PreprocessConfig, RuntimeSchema},
    checkpoint,
    config::ModelConfig,
    models::{
        lewm::{LeWm, LeWmConfig},
        tdmpc2::{TdMpc2, TdMpc2Config},
    },
    planner::{
        ActionBounds, CemConfig, CemPlanner, IcemConfig, IcemPlanner, LeWmGoalScorer, MppiConfig,
        MppiPlanner,
    },
    runtime::{DTypeSpec, DeviceSpec},
    session::{LeWmSession, TdMpc2Session},
};

type FfiResult<T> = std::result::Result<T, FfiError>;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwmStatus {
    Ok = 0,
    NullPointer = 1,
    InvalidArgument = 2,
    RuntimeError = 3,
    Panic = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwmCemPlanConfig {
    pub horizon: usize,
    pub samples: usize,
    pub elites: usize,
    pub iterations: usize,
    pub init_std: f32,
    pub min_std: f32,
}

impl Default for SwmCemPlanConfig {
    fn default() -> Self {
        Self {
            horizon: 5,
            samples: 512,
            elites: 64,
            iterations: 4,
            init_std: 1.0,
            min_std: 1e-3,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwmMppiPlanConfig {
    pub horizon: usize,
    pub samples: usize,
    pub iterations: usize,
    pub noise_std: f32,
    pub temperature: f32,
}

impl Default for SwmMppiPlanConfig {
    fn default() -> Self {
        Self {
            horizon: 5,
            samples: 512,
            iterations: 1,
            noise_std: 1.0,
            temperature: 1.0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SwmIcemPlanConfig {
    pub horizon: usize,
    pub samples: usize,
    pub elites: usize,
    pub keep_elites: usize,
    pub iterations: usize,
    pub init_std: f32,
    pub min_std: f32,
}

impl Default for SwmIcemPlanConfig {
    fn default() -> Self {
        Self {
            horizon: 5,
            samples: 512,
            elites: 64,
            keep_elites: 64,
            iterations: 4,
            init_std: 1.0,
            min_std: 1e-3,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SwmPixelLayout {
    Nchw = 0,
    Nhwc = 1,
}

pub struct SwmTdMpc2 {
    session: TdMpc2Session,
    state_dim: Option<usize>,
    image_size: Option<usize>,
    action_dim: usize,
    action_bounds: ActionBounds,
    icem_planner: Option<IcemPlanner>,
}

pub struct SwmLeWm {
    session: LeWmSession,
    goal_emb: Option<Tensor>,
    action_dim: usize,
    image_size: usize,
    history_size: usize,
    action_bounds: ActionBounds,
    icem_planner: Option<IcemPlanner>,
}

#[unsafe(no_mangle)]
pub extern "C" fn swm_last_error_message() -> *const c_char {
    LAST_ERROR.with(|last| {
        last.borrow()
            .as_ref()
            .map_or(ptr::null(), |message| message.as_ptr())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_load(
    artifact_dir: *const c_char,
    device: *const c_char,
    dtype: *const c_char,
    out: *mut *mut SwmTdMpc2,
) -> SwmStatus {
    ffi_guard(|| {
        let artifact_dir = unsafe { required_string(artifact_dir, "artifact_dir")? };
        let device = unsafe { optional_string(device)? }
            .as_deref()
            .unwrap_or("cpu")
            .parse::<DeviceSpec>()
            .map_err(FfiError::invalid)?
            .resolve()
            .map_err(FfiError::runtime)?;
        let dtype = unsafe { optional_string(dtype)? }
            .as_deref()
            .unwrap_or("f32")
            .parse::<DTypeSpec>()
            .map_err(FfiError::invalid)?
            .dtype();
        let out = unsafe { required_mut(out, "out")? };

        let artifact = DeploymentArtifact::from_dir(&artifact_dir).map_err(FfiError::runtime)?;
        let ModelConfig::TdMpc2(config) = artifact.config.clone() else {
            return Err(FfiError::invalid(
                "swm_tdmpc2_load only supports tdmpc2 artifacts",
            ));
        };
        let (state_dim, image_size) = tdmpc2_observation_dims(&config)?;
        let action_dim = config.action_dim;
        let action_bounds =
            action_bounds_from_schema(&artifact.schema, &artifact.preprocess, action_dim)?;
        let session = load_tdmpc2_session(config, &artifact.weights, dtype, &device)?;

        let handle = Box::new(SwmTdMpc2 {
            session,
            state_dim,
            image_size,
            action_dim,
            action_bounds,
            icem_planner: None,
        });
        *out = Box::into_raw(handle);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_load(
    artifact_dir: *const c_char,
    device: *const c_char,
    dtype: *const c_char,
    out: *mut *mut SwmLeWm,
) -> SwmStatus {
    ffi_guard(|| {
        let artifact_dir = unsafe { required_string(artifact_dir, "artifact_dir")? };
        let device = unsafe { optional_string(device)? }
            .as_deref()
            .unwrap_or("cpu")
            .parse::<DeviceSpec>()
            .map_err(FfiError::invalid)?
            .resolve()
            .map_err(FfiError::runtime)?;
        let dtype = unsafe { optional_string(dtype)? }
            .as_deref()
            .unwrap_or("f32")
            .parse::<DTypeSpec>()
            .map_err(FfiError::invalid)?
            .dtype();
        let out = unsafe { required_mut(out, "out")? };

        let artifact = DeploymentArtifact::from_dir(&artifact_dir).map_err(FfiError::runtime)?;
        let ModelConfig::LeWm(config) = artifact.config.clone() else {
            return Err(FfiError::invalid(
                "swm_lewm_load only supports le_wm artifacts",
            ));
        };
        let action_dim = config.action_encoder.input_dim;
        let image_size = config.encoder.image_size;
        let history_size = config.history_size;
        let action_bounds =
            action_bounds_from_schema(&artifact.schema, &artifact.preprocess, action_dim)?;
        let session = load_lewm_session(config, &artifact.weights, dtype, &device)?;

        let handle = Box::new(SwmLeWm {
            session,
            goal_emb: None,
            action_dim,
            image_size,
            history_size,
            action_bounds,
            icem_planner: None,
        });
        *out = Box::into_raw(handle);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_free(handle: *mut SwmTdMpc2) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_free(handle: *mut SwmLeWm) {
    if !handle.is_null() {
        unsafe {
            drop(Box::from_raw(handle));
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_state_dim(
    handle: *const SwmTdMpc2,
    out: *mut usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.state_dim.unwrap_or(0);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_image_size(
    handle: *const SwmTdMpc2,
    out: *mut usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.image_size.unwrap_or(0);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_action_dim(
    handle: *const SwmTdMpc2,
    out: *mut usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.action_dim;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_action_dim(handle: *const SwmLeWm, out: *mut usize) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.action_dim;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_image_size(handle: *const SwmLeWm, out: *mut usize) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.image_size;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_history_size(
    handle: *const SwmLeWm,
    out: *mut usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_ref(handle, "handle")? };
        let out = unsafe { required_mut(out, "out")? };
        *out = handle.history_size;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_reset_state(
    handle: *mut SwmTdMpc2,
    state: *const f32,
    batch: usize,
    state_dim: usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        if handle.state_dim.is_some() && handle.image_size.is_some() {
            return Err(FfiError::invalid(
                "TD-MPC2 artifact also requires pixels; use swm_tdmpc2_reset_state_pixels",
            ));
        }
        let state = unsafe { state_tensor_from_ffi(handle, state, batch, state_dim)? };
        handle
            .session
            .reset_state(&state)
            .map_err(FfiError::runtime)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_reset_pixels(
    handle: *mut SwmLeWm,
    pixels: *const f32,
    batch: usize,
    time: usize,
    height: usize,
    width: usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        if time != handle.history_size {
            return Err(FfiError::invalid(format!(
                "LeWM current pixel history must have time={} frames, got {time}",
                handle.history_size
            )));
        }
        let pixels =
            unsafe { lewm_pixels_from_ffi(handle, pixels, batch, time, height, width, "pixels")? };
        handle
            .session
            .reset_pixels(&pixels)
            .map_err(FfiError::runtime)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_set_goal_pixels(
    handle: *mut SwmLeWm,
    pixels: *const f32,
    batch: usize,
    time: usize,
    height: usize,
    width: usize,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let pixels = unsafe {
            lewm_pixels_from_ffi(handle, pixels, batch, time, height, width, "goal_pixels")?
        };
        let goal_emb = handle
            .session
            .encode_pixels(&pixels)
            .map_err(FfiError::runtime)?;
        handle.goal_emb = Some(goal_emb);
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_reset_pixels(
    handle: *mut SwmTdMpc2,
    pixels: *const f32,
    batch: usize,
    height: usize,
    width: usize,
    layout: c_int,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        if handle.state_dim.is_some() && handle.image_size.is_some() {
            return Err(FfiError::invalid(
                "TD-MPC2 artifact also requires state; use swm_tdmpc2_reset_state_pixels",
            ));
        }
        let pixels =
            unsafe { pixel_tensor_from_ffi(handle, pixels, batch, height, width, layout)? };
        handle
            .session
            .reset_pixels(&pixels)
            .map_err(FfiError::runtime)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_reset_state_pixels(
    handle: *mut SwmTdMpc2,
    state: *const f32,
    pixels: *const f32,
    batch: usize,
    state_dim: usize,
    height: usize,
    width: usize,
    layout: c_int,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let state = unsafe { state_tensor_from_ffi(handle, state, batch, state_dim)? };
        let pixels =
            unsafe { pixel_tensor_from_ffi(handle, pixels, batch, height, width, layout)? };
        handle
            .session
            .reset_observations(&[("pixels", &pixels), ("state", &state)])
            .map_err(FfiError::runtime)?;
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_plan_cem(
    handle: *mut SwmTdMpc2,
    ffi_config: SwmCemPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let mut config = CemConfig::new(
            ffi_config.horizon,
            ffi_config.samples,
            ffi_config.elites,
            handle.action_dim,
        );
        config.iterations = ffi_config.iterations;
        config.init_std = ffi_config.init_std;
        config.min_std = ffi_config.min_std;
        config.action_bounds = handle.action_bounds.clone();

        let result = CemPlanner::new(config)
            .plan(&handle.session)
            .map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_plan_mppi(
    handle: *mut SwmTdMpc2,
    ffi_config: SwmMppiPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let mut config = MppiConfig::new(ffi_config.horizon, ffi_config.samples, handle.action_dim);
        config.iterations = ffi_config.iterations;
        config.noise_std = ffi_config.noise_std;
        config.temperature = ffi_config.temperature;
        config.action_bounds = handle.action_bounds.clone();

        let result = MppiPlanner::new(config)
            .plan(&handle.session)
            .map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_plan_icem(
    handle: *mut SwmTdMpc2,
    ffi_config: SwmIcemPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let config = icem_config_from_ffi(ffi_config, handle.action_dim, &handle.action_bounds);
        let planner = match handle.icem_planner.as_mut() {
            Some(planner) if planner.config() == &config => planner,
            _ => {
                handle.icem_planner = Some(IcemPlanner::new(config));
                handle.icem_planner.as_mut().expect("iCEM planner set")
            }
        };

        let result = planner.plan(&handle.session).map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_plan_cem(
    handle: *mut SwmLeWm,
    ffi_config: SwmCemPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let goal_emb = handle
            .goal_emb
            .as_ref()
            .ok_or_else(|| FfiError::invalid("LeWM goal pixels must be set before planning"))?;
        let mut config = CemConfig::new(
            ffi_config.horizon,
            ffi_config.samples,
            ffi_config.elites,
            handle.action_dim,
        );
        config.iterations = ffi_config.iterations;
        config.init_std = ffi_config.init_std;
        config.min_std = ffi_config.min_std;
        config.action_bounds = handle.action_bounds.clone();

        let scorer = LeWmGoalScorer::new(&handle.session, goal_emb);
        let result = CemPlanner::new(config)
            .plan(&scorer)
            .map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_plan_mppi(
    handle: *mut SwmLeWm,
    ffi_config: SwmMppiPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let goal_emb = handle
            .goal_emb
            .as_ref()
            .ok_or_else(|| FfiError::invalid("LeWM goal pixels must be set before planning"))?;
        let mut config = MppiConfig::new(ffi_config.horizon, ffi_config.samples, handle.action_dim);
        config.iterations = ffi_config.iterations;
        config.noise_std = ffi_config.noise_std;
        config.temperature = ffi_config.temperature;
        config.action_bounds = handle.action_bounds.clone();

        let scorer = LeWmGoalScorer::new(&handle.session, goal_emb);
        let result = MppiPlanner::new(config)
            .plan(&scorer)
            .map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_plan_icem(
    handle: *mut SwmLeWm,
    ffi_config: SwmIcemPlanConfig,
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        let goal_emb = handle
            .goal_emb
            .as_ref()
            .ok_or_else(|| FfiError::invalid("LeWM goal pixels must be set before planning"))?;
        let config = icem_config_from_ffi(ffi_config, handle.action_dim, &handle.action_bounds);
        let planner = match handle.icem_planner.as_mut() {
            Some(planner) if planner.config() == &config => planner,
            _ => {
                handle.icem_planner = Some(IcemPlanner::new(config));
                handle.icem_planner.as_mut().expect("iCEM planner set")
            }
        };

        let scorer = LeWmGoalScorer::new(&handle.session, goal_emb);
        let result = planner.plan(&scorer).map_err(FfiError::runtime)?;
        copy_plan_outputs(
            &result.first_action,
            &result.sequence,
            &result.scores,
            &result.best_indices,
            action_out,
            sequence_out,
            best_cost_out,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_tdmpc2_clear_icem_warm_start(handle: *mut SwmTdMpc2) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        if let Some(planner) = handle.icem_planner.as_mut() {
            planner.clear_warm_start();
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn swm_lewm_clear_icem_warm_start(handle: *mut SwmLeWm) -> SwmStatus {
    ffi_guard(|| {
        let handle = unsafe { required_mut(handle, "handle")? };
        if let Some(planner) = handle.icem_planner.as_mut() {
            planner.clear_warm_start();
        }
        Ok(())
    })
}

fn icem_config_from_ffi(
    ffi_config: SwmIcemPlanConfig,
    action_dim: usize,
    action_bounds: &ActionBounds,
) -> IcemConfig {
    let mut config = IcemConfig::new(
        ffi_config.horizon,
        ffi_config.samples,
        ffi_config.elites,
        action_dim,
    );
    config.keep_elites = ffi_config.keep_elites;
    config.iterations = ffi_config.iterations;
    config.init_std = ffi_config.init_std;
    config.min_std = ffi_config.min_std;
    config.action_bounds = action_bounds.clone();
    config
}

fn load_tdmpc2_session(
    config: TdMpc2Config,
    weights: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> FfiResult<TdMpc2Session> {
    let vb =
        checkpoint::var_builder_from_path(weights, dtype, device).map_err(FfiError::runtime)?;
    let model = TdMpc2::new(config, vb).map_err(FfiError::runtime)?;
    Ok(TdMpc2Session::new(model, device.clone(), dtype))
}

fn load_lewm_session(
    config: LeWmConfig,
    weights: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> FfiResult<LeWmSession> {
    let vb =
        checkpoint::var_builder_from_path(weights, dtype, device).map_err(FfiError::runtime)?;
    let model = LeWm::new(config, vb).map_err(FfiError::runtime)?;
    Ok(LeWmSession::new(model, device.clone(), dtype))
}

unsafe fn lewm_pixels_from_ffi(
    handle: &SwmLeWm,
    pixels: *const f32,
    batch: usize,
    time: usize,
    height: usize,
    width: usize,
    name: &str,
) -> FfiResult<Tensor> {
    if batch == 0 || time == 0 {
        return Err(FfiError::invalid(
            "LeWM pixel batch and time must be greater than zero",
        ));
    }
    if height != handle.image_size || width != handle.image_size {
        return Err(FfiError::invalid(format!(
            "LeWM pixel input must match image_size {}, got {height}x{width}",
            handle.image_size
        )));
    }
    let len = batch
        .checked_mul(time)
        .and_then(|len| len.checked_mul(3))
        .and_then(|len| len.checked_mul(height))
        .and_then(|len| len.checked_mul(width))
        .ok_or_else(|| FfiError::invalid("LeWM pixel length overflow"))?;
    let pixels = unsafe { required_slice(pixels, len, name)? };
    Tensor::from_slice(
        pixels,
        (batch, time, 3usize, height, width),
        handle.session.device(),
    )
    .map_err(FfiError::runtime)
}

unsafe fn state_tensor_from_ffi(
    handle: &SwmTdMpc2,
    state: *const f32,
    batch: usize,
    state_dim: usize,
) -> FfiResult<Tensor> {
    if batch == 0 {
        return Err(FfiError::invalid("batch must be greater than zero"));
    }
    let Some(expected_state_dim) = handle.state_dim else {
        return Err(FfiError::invalid(
            "TD-MPC2 artifact does not have a state observation",
        ));
    };
    if state_dim != expected_state_dim {
        return Err(FfiError::invalid(format!(
            "state_dim {state_dim} does not match runtime state_dim {expected_state_dim}"
        )));
    }
    let len = batch
        .checked_mul(state_dim)
        .ok_or_else(|| FfiError::invalid("state length overflow"))?;
    let state = unsafe { required_slice(state, len, "state")? };
    Tensor::from_slice(state, (batch, state_dim), handle.session.device())
        .map_err(FfiError::runtime)
}

unsafe fn pixel_tensor_from_ffi(
    handle: &SwmTdMpc2,
    pixels: *const f32,
    batch: usize,
    height: usize,
    width: usize,
    layout: c_int,
) -> FfiResult<Tensor> {
    if batch == 0 {
        return Err(FfiError::invalid("batch must be greater than zero"));
    }
    let Some(image_size) = handle.image_size else {
        return Err(FfiError::invalid(
            "TD-MPC2 artifact does not have a pixel observation",
        ));
    };
    if height != image_size || width != image_size {
        return Err(FfiError::invalid(format!(
            "pixel input must match image_size {image_size}, got {height}x{width}"
        )));
    }
    let len = batch
        .checked_mul(3)
        .and_then(|len| len.checked_mul(height))
        .and_then(|len| len.checked_mul(width))
        .ok_or_else(|| FfiError::invalid("pixel length overflow"))?;
    let pixels = unsafe { required_slice(pixels, len, "pixels")? };
    match parse_pixel_layout(layout)? {
        SwmPixelLayout::Nchw => Tensor::from_slice(
            pixels,
            (batch, 3usize, height, width),
            handle.session.device(),
        ),
        SwmPixelLayout::Nhwc => Tensor::from_slice(
            pixels,
            (batch, height, width, 3usize),
            handle.session.device(),
        ),
    }
    .map_err(FfiError::runtime)
}

fn parse_pixel_layout(layout: c_int) -> FfiResult<SwmPixelLayout> {
    match layout {
        0 => Ok(SwmPixelLayout::Nchw),
        1 => Ok(SwmPixelLayout::Nhwc),
        other => Err(FfiError::invalid(format!(
            "unknown pixel layout {other}; expected 0=NCHW or 1=NHWC"
        ))),
    }
}

fn tdmpc2_observation_dims(config: &TdMpc2Config) -> FfiResult<(Option<usize>, Option<usize>)> {
    let mut state_dim = None;
    let mut image_size = None;

    for encoding in &config.encodings {
        match encoding.name.as_str() {
            "state" => {
                if state_dim.replace(encoding.input_dim).is_some() {
                    return Err(FfiError::invalid(
                        "TD-MPC2 C ABI does not support duplicate state observations",
                    ));
                }
            }
            "pixels" => {
                let size = config.image_size.unwrap_or(encoding.input_dim);
                if image_size.replace(size).is_some() {
                    return Err(FfiError::invalid(
                        "TD-MPC2 C ABI does not support duplicate pixel observations",
                    ));
                }
            }
            other => {
                return Err(FfiError::invalid(format!(
                    "TD-MPC2 C ABI does not support observation '{other}'"
                )));
            }
        }
    }

    if state_dim.is_none() && image_size.is_none() {
        return Err(FfiError::invalid(
            "TD-MPC2 C ABI requires a state or pixel observation",
        ));
    }

    Ok((state_dim, image_size))
}

fn action_bounds_from_schema(
    schema: &RuntimeSchema,
    preprocess: &PreprocessConfig,
    action_dim: usize,
) -> FfiResult<ActionBounds> {
    let low = schema
        .action
        .min
        .clone()
        .or_else(|| preprocess.action_min.clone())
        .unwrap_or_else(|| vec![-1.0; action_dim]);
    let high = schema
        .action
        .max
        .clone()
        .or_else(|| preprocess.action_max.clone())
        .unwrap_or_else(|| vec![1.0; action_dim]);
    if low.len() != action_dim || high.len() != action_dim {
        return Err(FfiError::invalid(format!(
            "action bounds must match action_dim {action_dim}, got low={} high={}",
            low.len(),
            high.len()
        )));
    }
    Ok(ActionBounds { low, high })
}

fn copy_plan_outputs(
    first_action: &Tensor,
    sequence: &Tensor,
    scores: &Tensor,
    best_indices: &[usize],
    action_out: *mut f32,
    sequence_out: *mut f32,
    best_cost_out: *mut f32,
) -> FfiResult<()> {
    let action = flatten2(first_action)?;
    unsafe {
        copy_required(&action, action_out, "action_out")?;
    }

    if !sequence_out.is_null() {
        let sequence = flatten3(sequence)?;
        unsafe {
            copy_optional(&sequence, sequence_out);
        }
    }

    if !best_cost_out.is_null() {
        let scores = scores.to_vec2::<f32>().map_err(FfiError::runtime)?;
        let mut best_costs = Vec::with_capacity(scores.len());
        for (row, &best_idx) in scores.iter().zip(best_indices.iter()) {
            let Some(&cost) = row.get(best_idx) else {
                return Err(FfiError::runtime(format!(
                    "best index {best_idx} out of range for score row length {}",
                    row.len()
                )));
            };
            best_costs.push(cost);
        }
        unsafe {
            copy_optional(&best_costs, best_cost_out);
        }
    }

    Ok(())
}

fn flatten2(tensor: &Tensor) -> FfiResult<Vec<f32>> {
    Ok(tensor
        .to_vec2::<f32>()
        .map_err(FfiError::runtime)?
        .into_iter()
        .flatten()
        .collect())
}

fn flatten3(tensor: &Tensor) -> FfiResult<Vec<f32>> {
    Ok(tensor
        .to_vec3::<f32>()
        .map_err(FfiError::runtime)?
        .into_iter()
        .flatten()
        .flatten()
        .collect())
}

unsafe fn required_string(ptr: *const c_char, name: &str) -> FfiResult<String> {
    if ptr.is_null() {
        return Err(FfiError::null(format!("{name} pointer is null")));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(str::to_owned)
        .map_err(|err| FfiError::invalid(format!("{name} is not valid UTF-8: {err}")))
}

unsafe fn optional_string(ptr: *const c_char) -> FfiResult<Option<String>> {
    if ptr.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|value| Some(value.to_owned()))
        .map_err(|err| FfiError::invalid(format!("string is not valid UTF-8: {err}")))
}

unsafe fn required_ref<'a, T>(ptr: *const T, name: &str) -> FfiResult<&'a T> {
    if ptr.is_null() {
        return Err(FfiError::null(format!("{name} pointer is null")));
    }
    Ok(unsafe { &*ptr })
}

unsafe fn required_mut<'a, T>(ptr: *mut T, name: &str) -> FfiResult<&'a mut T> {
    if ptr.is_null() {
        return Err(FfiError::null(format!("{name} pointer is null")));
    }
    Ok(unsafe { &mut *ptr })
}

unsafe fn required_slice<'a, T>(ptr: *const T, len: usize, name: &str) -> FfiResult<&'a [T]> {
    if ptr.is_null() {
        return Err(FfiError::null(format!("{name} pointer is null")));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

unsafe fn copy_required(values: &[f32], out: *mut f32, name: &str) -> FfiResult<()> {
    if out.is_null() {
        return Err(FfiError::null(format!("{name} pointer is null")));
    }
    unsafe {
        copy_optional(values, out);
    }
    Ok(())
}

unsafe fn copy_optional(values: &[f32], out: *mut f32) {
    unsafe {
        ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    }
}

fn ffi_guard<F>(f: F) -> SwmStatus
where
    F: FnOnce() -> FfiResult<()>,
{
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => SwmStatus::Ok,
        Ok(Err(err)) => {
            let status = err.status;
            set_last_error(err.message);
            status
        }
        Err(_) => {
            set_last_error("panic crossed stable-worldmodel C ABI".to_string());
            SwmStatus::Panic
        }
    }
}

fn clear_last_error() {
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = None;
    });
}

fn set_last_error(message: String) {
    let message = message.replace('\0', "\\0");
    LAST_ERROR.with(|last| {
        *last.borrow_mut() = CString::new(message).ok();
    });
}

#[derive(Debug)]
struct FfiError {
    status: SwmStatus,
    message: String,
}

impl FfiError {
    fn null(message: impl Into<String>) -> Self {
        Self {
            status: SwmStatus::NullPointer,
            message: message.into(),
        }
    }

    fn invalid(message: impl std::fmt::Display) -> Self {
        Self {
            status: SwmStatus::InvalidArgument,
            message: message.to_string(),
        }
    }

    fn runtime(message: impl std::fmt::Display) -> Self {
        Self {
            status: SwmStatus::RuntimeError,
            message: message.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::tdmpc2::EncodingConfig;

    #[test]
    fn tdmpc2_observation_dims_support_state_pixel_and_mixed() -> FfiResult<()> {
        assert_eq!(
            tdmpc2_observation_dims(&TdMpc2Config::state_only(12, 4))?,
            (Some(12), None)
        );
        assert_eq!(
            tdmpc2_observation_dims(&TdMpc2Config::pixel_only(64, 4, 128))?,
            (None, Some(64))
        );

        let mut mixed = TdMpc2Config::pixel_only(64, 4, 128);
        mixed.encodings.push(EncodingConfig::new("state", 12, 128));
        assert_eq!(tdmpc2_observation_dims(&mixed)?, (Some(12), Some(64)));

        Ok(())
    }

    #[test]
    fn tdmpc2_observation_dims_reject_unknown_observation() {
        let mut config = TdMpc2Config::state_only(12, 4);
        config
            .encodings
            .push(EncodingConfig::new("proprioceptive", 6, 128));

        let err = tdmpc2_observation_dims(&config).unwrap_err();
        assert_eq!(err.status, SwmStatus::InvalidArgument);
        assert!(err.message.contains("proprioceptive"));
    }
}
