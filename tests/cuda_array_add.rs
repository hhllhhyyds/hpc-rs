use hpc_rs::add_array::add_array_cuda;
use serial_test::serial;

#[test]
#[serial]
fn test_array_add() {
    hpc_rs::device::cuda_device_reset();
    hpc_rs::device::cuda_set_device();
    {
        let out = add_array_cuda(&[1., 2., 3., 4.], &[1., 2., 3., 4.]);
        for (i, x) in out.into_iter().enumerate() {
            assert!((i + 1) as f32 * 2. == x, "i = {}, x = {}", i, x);
        }

        const N: usize = 10000000_usize;
        let a: Vec<_> = (0..N).map(|x| x as f32 / 10000.).collect();
        let b: Vec<_> = (0..N).rev().map(|x| x as f32 / 10000.).collect();
        let start = std::time::Instant::now();
        let out = add_array_cuda(&a, &b);
        let end = std::time::Instant::now();
        let duration = end - start;
        println!(
            "time used in add array = {} ms",
            duration.as_nanos() / 1000000
        );

        for (i, x) in out.into_iter().enumerate() {
            assert!(
                (x - (N - 1) as f32 / 10000.).abs() < 1e-3,
                "i = {}, x = {}, n - 1 = {}",
                i,
                x,
                (N - 1) as f32 / 10000.
            )
        }
    }
    hpc_rs::device::cuda_device_reset();
}
