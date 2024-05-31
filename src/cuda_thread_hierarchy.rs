#[derive(Clone, Copy, Debug)]
pub struct Dim3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct CudaThreadHierarchy {
    pub block: Dim3,
    pub grid: Dim3,
}

impl From<(usize, usize, usize)> for Dim3 {
    fn from(value: (usize, usize, usize)) -> Self {
        Self {
            x: value.0,
            y: value.1,
            z: value.2,
        }
    }
}

impl CudaThreadHierarchy {
    pub const DEFAULT_BLOCK_DIM_X: usize = 1024;

    pub fn dim1_default(n: usize) -> Self {
        Self {
            block: (Self::DEFAULT_BLOCK_DIM_X, 1, 1).into(),
            grid: (
                (n + Self::DEFAULT_BLOCK_DIM_X - 1) / Self::DEFAULT_BLOCK_DIM_X,
                1,
                1,
            )
                .into(),
        }
    }
}
