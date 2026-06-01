use std::{ffi::CString, ffi::c_void, ptr};

use stable_worldmodel_candle::ffi::{
    SwmCudaImage, SwmCudaNv12, SwmIcemPlanConfig, SwmLeWm, SwmStatus, SwmTdMpc2,
    swm_cuda_image_alloc, swm_cuda_image_free, swm_cuda_image_ptr, swm_cuda_nv12_alloc,
    swm_cuda_nv12_free, swm_cuda_nv12_uv_ptr, swm_cuda_nv12_y_ptr, swm_last_error_message,
    swm_lewm_clear_icem_warm_start, swm_lewm_free, swm_lewm_load, swm_lewm_plan_cem,
    swm_lewm_reset_cuda_image_history, swm_lewm_reset_pixels, swm_lewm_set_goal_pixels,
    swm_tdmpc2_clear_icem_warm_start, swm_tdmpc2_free, swm_tdmpc2_load, swm_tdmpc2_plan_icem,
    swm_tdmpc2_reset_cuda_image, swm_tdmpc2_reset_pixels, swm_tdmpc2_reset_state_pixels,
};

#[test]
fn ffi_load_rejects_null_artifact_path() {
    let mut handle: *mut SwmTdMpc2 = ptr::null_mut();
    let status = unsafe {
        swm_tdmpc2_load(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &mut handle as *mut *mut SwmTdMpc2,
        )
    };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(handle.is_null());
    assert!(last_error().contains("artifact_dir"));
}

#[test]
fn ffi_load_rejects_missing_artifact_dir() {
    let path = CString::new("/definitely/not/a/stable-worldmodel-artifact").unwrap();
    let mut handle: *mut SwmTdMpc2 = ptr::null_mut();
    let status = unsafe { swm_tdmpc2_load(path.as_ptr(), ptr::null(), ptr::null(), &mut handle) };

    assert_eq!(status, SwmStatus::RuntimeError);
    assert!(handle.is_null());
    assert!(last_error().contains("config.json"));
}

#[test]
fn ffi_lewm_load_rejects_null_artifact_path() {
    let mut handle: *mut SwmLeWm = ptr::null_mut();
    let status = unsafe {
        swm_lewm_load(
            ptr::null(),
            ptr::null(),
            ptr::null(),
            &mut handle as *mut *mut SwmLeWm,
        )
    };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(handle.is_null());
    assert!(last_error().contains("artifact_dir"));
}

#[test]
fn ffi_free_accepts_null() {
    unsafe {
        swm_tdmpc2_free(ptr::null_mut());
        swm_lewm_free(ptr::null_mut());
        swm_cuda_image_free(ptr::null_mut());
        swm_cuda_nv12_free(ptr::null_mut());
    }
}

#[test]
fn ffi_cuda_image_alloc_exposes_device_pointer() {
    let mut image: *mut SwmCudaImage = ptr::null_mut();
    let status =
        unsafe { swm_cuda_image_alloc(ptr::null(), 1, 2, 3, 0, &mut image as *mut *mut _) };

    assert_eq!(status, SwmStatus::Ok);
    assert!(!image.is_null());

    let mut data: *mut c_void = ptr::null_mut();
    let mut pitch = 0usize;
    let status = unsafe { swm_cuda_image_ptr(image, &mut data, &mut pitch) };

    assert_eq!(status, SwmStatus::Ok);
    assert!(!data.is_null());
    assert_eq!(pitch, 9);

    unsafe {
        swm_cuda_image_free(image);
    }
}

#[test]
fn ffi_cuda_nv12_alloc_exposes_plane_pointers() {
    let mut nv12: *mut SwmCudaNv12 = ptr::null_mut();
    let status = unsafe { swm_cuda_nv12_alloc(ptr::null(), 1, 4, 6, &mut nv12 as *mut *mut _) };

    assert_eq!(status, SwmStatus::Ok);
    assert!(!nv12.is_null());

    let mut y_ptr: *mut c_void = ptr::null_mut();
    let mut uv_ptr: *mut c_void = ptr::null_mut();
    let mut y_pitch = 0usize;
    let mut uv_pitch = 0usize;
    let y_status = unsafe { swm_cuda_nv12_y_ptr(nv12, &mut y_ptr, &mut y_pitch) };
    let uv_status = unsafe { swm_cuda_nv12_uv_ptr(nv12, &mut uv_ptr, &mut uv_pitch) };

    assert_eq!(y_status, SwmStatus::Ok);
    assert_eq!(uv_status, SwmStatus::Ok);
    assert!(!y_ptr.is_null());
    assert!(!uv_ptr.is_null());
    assert_eq!(y_pitch, 6);
    assert_eq!(uv_pitch, 6);

    unsafe {
        swm_cuda_nv12_free(nv12);
    }
}

#[test]
fn ffi_cuda_image_alloc_rejects_unknown_format() {
    let mut image: *mut SwmCudaImage = ptr::null_mut();
    let status =
        unsafe { swm_cuda_image_alloc(ptr::null(), 1, 2, 3, 99, &mut image as *mut *mut _) };

    assert_eq!(status, SwmStatus::RuntimeError);
    assert!(image.is_null());
    assert!(last_error().contains("unknown packed CUDA image format"));
}

#[test]
fn ffi_plan_icem_rejects_null_handle() {
    let mut action = [0f32; 4];
    let status = unsafe {
        swm_tdmpc2_plan_icem(
            ptr::null_mut(),
            SwmIcemPlanConfig::default(),
            action.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_clear_icem_warm_start_rejects_null_handle() {
    let status = unsafe { swm_tdmpc2_clear_icem_warm_start(ptr::null_mut()) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_reset_pixels_rejects_null_handle() {
    let pixels = [0f32; 3 * 4 * 4];
    let status = unsafe { swm_tdmpc2_reset_pixels(ptr::null_mut(), pixels.as_ptr(), 1, 4, 4, 0) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_reset_cuda_image_rejects_null_handle() {
    let mut image: *mut SwmCudaImage = ptr::null_mut();
    let status =
        unsafe { swm_cuda_image_alloc(ptr::null(), 1, 4, 4, 0, &mut image as *mut *mut _) };
    assert_eq!(status, SwmStatus::Ok);

    let status = unsafe { swm_tdmpc2_reset_cuda_image(ptr::null_mut(), image) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));

    unsafe {
        swm_cuda_image_free(image);
    }
}

#[test]
fn ffi_reset_state_pixels_rejects_null_handle() {
    let state = [0f32; 4];
    let pixels = [0f32; 3 * 4 * 4];
    let status = unsafe {
        swm_tdmpc2_reset_state_pixels(
            ptr::null_mut(),
            state.as_ptr(),
            pixels.as_ptr(),
            1,
            4,
            4,
            4,
            0,
        )
    };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_lewm_reset_pixels_rejects_null_handle() {
    let pixels = [0f32; 3 * 4 * 4];
    let status = unsafe { swm_lewm_reset_pixels(ptr::null_mut(), pixels.as_ptr(), 1, 1, 4, 4) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_lewm_reset_cuda_image_history_rejects_null_handle() {
    let mut image: *mut SwmCudaImage = ptr::null_mut();
    let status =
        unsafe { swm_cuda_image_alloc(ptr::null(), 3, 4, 4, 0, &mut image as *mut *mut _) };
    assert_eq!(status, SwmStatus::Ok);

    let status = unsafe { swm_lewm_reset_cuda_image_history(ptr::null_mut(), image, 1, 3) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));

    unsafe {
        swm_cuda_image_free(image);
    }
}

#[test]
fn ffi_lewm_set_goal_pixels_rejects_null_handle() {
    let pixels = [0f32; 3 * 4 * 4];
    let status = unsafe { swm_lewm_set_goal_pixels(ptr::null_mut(), pixels.as_ptr(), 1, 1, 4, 4) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_lewm_plan_cem_rejects_null_handle() {
    let mut action = [0f32; 4];
    let status = unsafe {
        swm_lewm_plan_cem(
            ptr::null_mut(),
            Default::default(),
            action.as_mut_ptr(),
            ptr::null_mut(),
            ptr::null_mut(),
        )
    };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

#[test]
fn ffi_lewm_clear_icem_warm_start_rejects_null_handle() {
    let status = unsafe { swm_lewm_clear_icem_warm_start(ptr::null_mut()) };

    assert_eq!(status, SwmStatus::NullPointer);
    assert!(last_error().contains("handle"));
}

fn last_error() -> String {
    let ptr = swm_last_error_message();
    assert!(!ptr.is_null());
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
