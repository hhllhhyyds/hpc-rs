use hpc_rs::add_array::add_array_cuda;

#[test]
fn test_array_add() {
    let out = add_array_cuda(&vec![1., 2., 3., 4.], &vec![1., 2., 3., 4.]);
    for (i, x) in out.into_iter().enumerate() {
        assert!((i + 1) as f32 * 2. == x, "i = {}, x = {}", i, x);
    }
}
