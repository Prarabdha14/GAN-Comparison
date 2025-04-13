fixed_noise = torch.randn(25, latent_dim).to(device)

for epoch in range(epochs):
    for batch_idx, (real_imgs, _) in enumerate(tqdm(train_loader)):
        real_imgs = real_imgs.to(device)

        # Ground truths
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        #train generator
        optimizer_G.zero_grad()
        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        #train discriminator
        optimizer_D.zero_grad()

        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

    # Log losses to TensorBoard
    writer.add_scalar("LS-GAN/Generator Loss", g_loss.item(), epoch)
    writer.add_scalar("LS-GAN/Discriminator Loss", d_loss.item(), epoch)

    # Log generated images
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        img_grid = make_grid(fake, normalize=True, value_range=(-1, 1))
        writer.add_image("LS-GAN/Generated Images", img_grid, global_step=epoch)

    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")
