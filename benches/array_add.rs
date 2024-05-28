use criterion::{criterion_group, criterion_main, Criterion};

use hpc_rs::add_array::add_array_cuda;

fn add_array_cpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert!(a.len() == b.len());

    let n = a.len();
    let mut out = vec![0.; n];
    for i in 0..n {
        out[i] = a[i] + b[i];
    }

    out
}

const N: usize = 100000000_usize;

fn bench_array_add_cuda(c: &mut Criterion) {
    let a: Vec<_> = (0..N).map(|x| x as f32 / 10000.).collect();
    let b: Vec<_> = (0..N).rev().map(|x| x as f32 / 10000.).collect();
    c.bench_function("cuda add array", |ben| ben.iter(|| add_array_cuda(&a, &b)));
}

fn bench_array_add_cpu(c: &mut Criterion) {
    let a: Vec<_> = (0..N).map(|x| x as f32 / 10000.).collect();
    let b: Vec<_> = (0..N).rev().map(|x| x as f32 / 10000.).collect();
    c.bench_function("cpu add array", |ben| ben.iter(|| add_array_cpu(&a, &b)));
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = bench_array_add_cuda, bench_array_add_cpu
}
criterion_main!(benches);
