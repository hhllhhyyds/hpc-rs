use std::{collections::HashMap, sync::Arc};

use cudarc::{
    driver::{CudaDevice, CudaFunction},
    nvrtc::Ptx,
};

pub struct PtxLoader {
    cuda_funcs: HashMap<String, CudaFunction>,
}

impl PtxLoader {
    pub fn new(dev: &Arc<CudaDevice>, ptx: &str, module: &str, functions: &[&'static str]) -> Self {
        let dev = dev.clone();
        dev.load_ptx(Ptx::from_src(ptx), module, functions)
            .expect("Fail to load ptx");

        let functions = functions.iter().map(|f| f.to_string()).collect::<Vec<_>>();
        let mut cuda_funcs = HashMap::new();

        for f in functions.iter() {
            cuda_funcs.insert(
                f.clone(),
                dev.get_func(module, f)
                    .unwrap_or_else(|| panic!("Fail to get function {} from ptx", f)),
            );
        }
        Self { cuda_funcs }
    }

    pub fn get_func(&self, func_name: &str) -> Option<CudaFunction> {
        self.cuda_funcs.get(func_name).cloned()
    }
}
