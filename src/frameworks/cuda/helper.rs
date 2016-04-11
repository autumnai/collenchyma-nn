//! Provides useful macros for easier NN implementation for CUDA/cuDNN.

/// Returns cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr<T>(x: &::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*const ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *const ::libc::c_void>(
        *try!(
            try!(
                x.get(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
}

/// Returns mutable cuDNN ready memory pointer from a SharedTensor.
pub unsafe fn receive_memory_ptr_mut<T>(x: &mut ::co::tensor::SharedTensor<T>, device: &::co::device::DeviceType) -> Result<*mut ::libc::c_void, ::co::plugin::Error> {
    Ok(::std::mem::transmute::<u64, *mut ::libc::c_void>(
        *try!(
            try!(
                x.get_mut(device).ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to resolve memory."))
            ).as_mut_cuda().ok_or(::co::plugin::Error::MissingMemoryForDevice("Unable to receive CUDA memory."))
        ).id_c()
    ))
}
