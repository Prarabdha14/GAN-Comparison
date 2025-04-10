from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='real_images',
    input2='generated_images_wgan',
    cuda=True,  # or False if on CPU
    isc=False,
    fid=True,
    kid=False
)

print("✅ FID Score (WGAN):", metrics['frechet_inception_distance'])
