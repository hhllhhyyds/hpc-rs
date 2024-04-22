use cudarc::driver::DriverError;

use crate::cuda_initialize::ErrorCudaInitialize;

#[derive(Debug)]
pub enum Error {
    CudaInitializeError(ErrorCudaInitialize),
    CudaDriverError(DriverError),
}

impl From<ErrorCudaInitialize> for Error {
    fn from(err: ErrorCudaInitialize) -> Self {
        Self::CudaInitializeError(err)
    }
}

impl From<DriverError> for Error {
    fn from(err: DriverError) -> Self {
        Self::CudaDriverError(err)
    }
}
