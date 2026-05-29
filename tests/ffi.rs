use std::{ffi::CString, ptr};

use stable_worldmodel_candle::ffi::{
    SwmStatus, SwmTdMpc2, swm_last_error_message, swm_tdmpc2_free, swm_tdmpc2_load,
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

fn last_error() -> String {
    let ptr = swm_last_error_message();
    assert!(!ptr.is_null());
    unsafe { std::ffi::CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
