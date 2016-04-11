extern crate collenchyma_nn as co_nn;
extern crate collenchyma as co;

use std::iter::repeat;

use co::prelude::*;
use co_nn::*;
use co::plugin::numeric_helpers::{cast, Float};

pub fn get_native_backend() -> Backend<Native> {
    Backend::<Native>::default().unwrap()
}

pub fn write_to_memory<T: Copy>(mem: &mut MemoryType, data: &[T]) {
    match mem {
        &mut MemoryType::Native(ref mut mem) => {
            let mut mem_buffer = mem.as_mut_slice::<T>();
            for (index, datum) in data.iter().enumerate() {
                mem_buffer[index] = *datum;
            }
        },
        #[cfg(any(feature = "opencl", feature = "cuda"))]
        _ => {}
    }
}

pub fn write_to<T, B, C>(tensor: &mut SharedTensor<T>,
                         payload: &[T],
                         backend: Option<&Backend<B>>,
                         native: &Backend<C>)
    where T: Copy,
          B: IFramework + Clone,
          C: IFramework + Clone,
{

    let _ = tensor.add_device(native.device());
    tensor.sync(native.device()).unwrap();
    write_to_memory(tensor.get_mut(native.device()).unwrap(), payload);
    if let Some(backend) = backend {
        let _ = tensor.add_device(backend.device());
        tensor.sync(backend.device()).unwrap();
    } else {
        tensor.sync(native.device()).unwrap();
    }
}

pub fn get_memory<T, B, C>(backend: Option<&Backend<B>>,
                           native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>,
                                                    SharedTensor<T>, SharedTensor<u8>)
    where T: Float,
          B: IFramework + Clone,
          C: IFramework + Clone,
{
    let val = cast::<f64, T>(1.0).unwrap();
    let val2 = cast::<f64, T>(2.0).unwrap();
    let batch = 4;
    let w1 = 9;
    let h1 = 9;
    let d1 = 3;
    let k = 6;
    let f = 3;
    let w2 = (w1 - f + 0) / 1 + 1;
    let h2 = (h1 - f + 0) / 1 + 1;
    let mut x = SharedTensor::<T>::new(native.device(), &(batch, d1, h1, w1)).unwrap();
    let mut payload: &mut [T] = &mut repeat(val).take(x.capacity()).collect::<Vec<T>>();
    payload[0] = val2;

    write_to(&mut x, payload, backend, native);

    let mut filter = SharedTensor::<T>::new(native.device(), &(k, d1, f, f)).unwrap();
    let payload: &[T] = &repeat(val).take(filter.capacity()).collect::<Vec<T>>();

    write_to(&mut filter, payload, backend, native);

    let mut result = SharedTensor::<T>::new(native.device(), &(batch, k, h2, w2)).unwrap();
    let payload: &[T] = &repeat(val2).take(result.capacity()).collect::<Vec<T>>();

    write_to(&mut result, payload, backend, native);

    let workspace = if let Some(cuda) = backend {
        SharedTensor::<u8>::new(cuda.device(), &(4)).unwrap()
    } else {
        SharedTensor::<u8>::new(native.device(), &(4)).unwrap()
    };

    (x, result, filter, workspace)
}

#[allow(dead_code)]
pub fn get_grad_memory<T, B, C>(backend: Option<&Backend<B>>,
                                native: &Backend<C>) -> (SharedTensor<T>, SharedTensor<T>,
                                                         SharedTensor<T>, SharedTensor<T>,
                                                         SharedTensor<T>)
    where T: Float,
          B: IFramework + Clone,
          C: IFramework + Clone,
{
    let val = cast::<f64, T>(1f64).unwrap();
    let val2 = cast::<f64, T>(2f64).unwrap();
    let batch = 4;
    let w1 = 9;
    let h1 = 9;
    let d1 = 3;
    let k = 6;
    let f = 3;
    let w2 = (w1 - f + 0) / 1 + 1;
    let h2 = (h1 - f + 0) / 1 + 1;

    let mut x = SharedTensor::<T>::new(native.device(), &(batch, d1, h1, w1)).unwrap();
    let mut payload: &mut [T] = &mut repeat(val).take(x.capacity()).collect::<Vec<T>>();
    payload[0] = val2;

    write_to(&mut x, payload, backend, native);

    let mut x_diff = SharedTensor::<T>::new(native.device(), &(batch, k, h2, w2)).unwrap();
    let mut payload: &mut [T] = &mut repeat(val).take(x_diff.capacity()).collect::<Vec<T>>();
    payload[0] = val2;

    write_to(&mut x_diff, payload, backend, native);

    let mut filter = SharedTensor::<T>::new(native.device(), &(k, d1, f, f)).unwrap();
    let payload: &[T] = &repeat(val).take(filter.capacity()).collect::<Vec<T>>();

    write_to(&mut filter, payload, backend, native);

    let mut result = SharedTensor::<T>::new(native.device(), &(batch, k, h2, w2)).unwrap();
    let payload: &[T] = &repeat(val).take(result.capacity()).collect::<Vec<T>>();

    write_to(&mut result, payload, backend, native);

    let mut result_diff = SharedTensor::<T>::new(native.device(), &(batch, k, h2, w2)).unwrap();
    if let Some(cuda) = backend {
        result_diff.add_device(cuda.device()).unwrap();
    }

    (x, x_diff, result, result_diff, filter)
}

pub fn create_conv_config<T, B>(backend: &B, x: &SharedTensor<T>, result: &SharedTensor<T>,
                                filter: &mut SharedTensor<T>) -> Result<<B as co_nn::NN<T>>::CC,
                                                                        co::error::Error>
    where B: co_nn::Convolution<T>,
{
    backend.new_convolution_config(x, result, filter,
                                   ConvForwardAlgo::ImplicitGEMM,
                                   ConvBackwardFilterAlgo::ImplicitGEMM,
                                   ConvBackwardDataAlgo::ImplicitGEMM,
                                   &vec!(1,1), &vec!(0,0))
}

pub fn check_conv<T>(device: &DeviceType, mut result: SharedTensor<T>)
    where T: Float + ::std::fmt::Debug,
{
    use std::iter::repeat;

    result.sync(device).unwrap();

    let mem = result
        .get(device)
        .unwrap()
        .as_native()
        .unwrap();
    let mut payload: &mut [T] = &mut repeat(cast::<f64, T>(27.0f64).unwrap())
        .take(result.capacity())
        .collect::<Vec<T>>();

    let desc = result.desc();
    for i in 0..desc[desc.len() - 2] - 1 {
        let idx = i * desc[desc.len() - 1] * desc[desc.len() - 2];
        println!("payload offset @ i = {:?}: {:?}",
                 i, idx);
        payload[idx] =
            cast::<f64, T>(28.0).unwrap();
    }

    /*for (i, (v1, v2)) in mem.as_slice::<f32>().iter().zip(payload.iter()).enumerate() {
        assert!(*v1 == *v2, "i = {:?}", i);
    }*/

    assert_eq!(payload, mem.as_slice::<T>());
}

#[cfg(test)]
#[cfg(feature = "cuda")]
mod convolution_spec_cuda {

    use super::*;

    use co::prelude::*;
    use co_nn::*;
    use co::plugin::numeric_helpers::{Float};

    pub fn get_cuda_backend() -> Backend<Cuda> {
        Backend::<Cuda>::default().unwrap()
    }


    fn convolution<T>(plain: bool)
        where T: Float + ::std::fmt::Debug + frameworks::cuda::DataTypeInfo,
    {
        let backend = get_cuda_backend();
        let native = get_native_backend();
        let (mut x, mut result, mut filter, mut workspace) =
            get_memory::<T, Cuda, Native>(Some(&backend), &native);

        let conf = create_conv_config(&backend, &x, &result, &mut filter)
            .unwrap();
        if !plain {
            backend.convolution(&mut filter, &mut x, &mut result,
                                &mut workspace, &conf)
                .unwrap();
        } else {
            backend.convolution_plain(&mut filter, &mut x, &mut result,
                                      &mut workspace, &conf)
                .unwrap();
        }

        check_conv(native.device(), result);
    }

    #[test]
    fn convolution_f32() {
        convolution::<f32>(false);
    }

    #[test]
    fn convolution_f64() {
        convolution::<f64>(false);
    }

    #[test]
    fn unsynced_convolution_f32() {
        convolution::<f32>(true);
    }

    #[test]
    fn unsynced_convolution_f64() {
        convolution::<f64>(true);
    }

    /*
    #[test]
    fn it_computes_correct_convolution_grad_on_cuda_for_f32() {
    let backend = get_cuda_backend();
    let native = get_native_backend();
    let (mut x, mut x_diff, mut result, mut result_diff, mut filter) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

    let conf = backend.new_convolution_config(&x, &result, &mut filter, &vec!(1,1), &vec!(0,0)).unwrap();
    match backend.convolution_grad(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
    Ok(_) => {
    result_diff.sync(native.device()).unwrap();
    if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
    assert_eq!(&[0f32, 0f32, -6f32], mem.as_slice::<f32>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f64() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f64, 0f64, -6f64], mem.as_slice::<f64>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f32_plain() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f32, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f32, 0f32, -6f32], mem.as_slice::<f32>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}

#[test]
fn it_computes_correct_convolution_grad_on_cuda_for_f64_plain() {
let backend = get_cuda_backend();
let native = get_native_backend();
let (mut x, mut x_diff, mut result, mut result_diff, filter, conv) = get_grad_memory::<f64, Cuda, Native>(&backend, &native);

let conf = backend.new_convolution_config(&x, &result, &filter, &conv).unwrap();
match backend.convolution_grad_plain(&mut x, &mut x_diff, &mut result, &mut result_diff, &conf) {
Ok(_) => {
result_diff.sync(native.device()).unwrap();
if let Some(mem) = result_diff.get(native.device()).unwrap().as_native() {
assert_eq!(&[0f64, 0f64, -6f64], mem.as_slice::<f64>());
}
},
Err(err) => { println!("{:?}", err); assert!(false) }
}
}
*/
}

#[cfg(test)]
#[cfg(feature = "native")]
mod convolution_spec_native {
    use super::*;

    use co::backend::{IBackend};
    use co::frameworks::Native;
    use co_nn::*;
    use co::plugin::numeric_helpers::{Float};

    fn convolution<T>(plain: bool)
        where T: Float + ::std::fmt::Debug + frameworks::cuda::DataTypeInfo + Default,
    {
        let native = super::get_native_backend();
        let (mut x, mut result, mut filter, mut workspace) =
            get_memory::<T, Native, Native>(None, &native);

        let conf = create_conv_config(&native, &x, &result, &mut filter)
            .unwrap();
        if !plain {
            native.convolution(&mut filter, &mut x, &mut result,
                               &mut workspace, &conf)
                .unwrap();
        } else {
            native.convolution_plain(&mut filter, &mut x, &mut result,
                                     &mut workspace, &conf)
                .unwrap();
        }

        check_conv(native.device(), result);
    }

    #[test]
    fn convolution_f32() {
        convolution::<f32>(false);
    }

    #[test]
    fn convolution_f64() {
        convolution::<f64>(false);
    }

    #[test]
    fn unsynced_convolution_f32() {
        convolution::<f32>(true);
    }

    #[test]
    fn unsynced_convolution_f64() {
        convolution::<f64>(true);
    }
}
