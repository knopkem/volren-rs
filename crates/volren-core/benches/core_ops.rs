use criterion::{black_box, criterion_group, criterion_main, Criterion};
use glam::{DMat3, DVec3, UVec3};
use volren_core::{
    transfer_function::{ColorTransferFunction, OpacityTransferFunction, TransferFunctionLut},
    Volume,
};

fn bench_sample_linear(c: &mut Criterion) {
    let data: Vec<u16> = (0..64 * 64 * 64)
        .map(|index| (index % 4096) as u16)
        .collect();
    let volume = Volume::from_data(
        data,
        UVec3::new(64, 64, 64),
        DVec3::ONE,
        DVec3::ZERO,
        DMat3::IDENTITY,
        1,
    )
    .expect("valid benchmark volume");

    c.bench_function("volume_sample_linear", |b| {
        b.iter(|| {
            black_box(
                volume
                    .sample_linear(DVec3::new(21.25, 33.5, 18.75))
                    .expect("in bounds"),
            )
        })
    });
}

fn bench_lut_bake(c: &mut Criterion) {
    let mut color_tf = ColorTransferFunction::greyscale(-1024.0, 3071.0);
    color_tf.add_point(300.0, [1.0, 0.8, 0.7]);
    let mut opacity_tf = OpacityTransferFunction::linear_ramp(-1024.0, 3071.0);
    opacity_tf.add_point(150.0, 0.2);
    opacity_tf.add_point(600.0, 0.8);

    c.bench_function("transfer_function_lut_bake", |b| {
        b.iter(|| {
            black_box(TransferFunctionLut::bake(
                black_box(&color_tf),
                black_box(&opacity_tf),
                -1024.0,
                3071.0,
                4096,
            ))
        })
    });
}

criterion_group!(benches, bench_sample_linear, bench_lut_bake);
criterion_main!(benches);
