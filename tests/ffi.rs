use std::{ffi::CString, ptr};

use stable_worldmodel_candle::ffi::{
    SwmIcemPlanConfig, SwmStatus, SwmTdMpc2, swm_last_error_message,
    swm_tdmpc2_clear_icem_warm_start, swm_tdmpc2_free, swm_tdmpc2_load, swm_tdmpc2_plan_icem,
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
fn ffi_free_accepts_null() {
    unsafe {
        swm_tdmpc2_free(ptr::null_mut());
    }
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

fn last_error() -> String {
    let ptr = swm_last_error_message();
    assert!(!ptr.is_null());
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
