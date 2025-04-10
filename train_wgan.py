for epoch in range(epochs):
    for i, (real_imgs, _) in enumerate(tqdm(train_loader)):
        real_imgs = real_imgs.to(device)

        # Train Critic
        for _ in range(n_critic):
            optimizer_C.zero_grad()

            # Generate fake images
            z = torch.randn(real_imgs.size(0), latent_dim).to(device)
            fake_imgs = generator(z).detach()

            # Critic loss (Wasserstein loss)
            loss_critic = -torch.mean(critic(real_imgs)) + torch.mean(critic(fake_imgs))

            loss_critic.backward()
            optimizer_C.step()

            # Weight clipping
            for p in critic.parameters():
                p.data.clamp_(-clip_value, clip_value)

        # Train Generator
        optimizer_G.zero_grad()

        z = torch.randn(real_imgs.size(0), latent_dim).to(device)
        gen_imgs = generator(z)

        # Generator loss
        loss_generator = -torch.mean(critic(gen_imgs))

        loss_generator.backward()
        optimizer_G.step()

    # TensorBoard Logging
    writer.add_scalar("WGAN/Generator Loss", loss_generator.item(), epoch)
    writer.add_scalar("WGAN/Critic Loss", loss_critic.item(), epoch)

    # Log generated images
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        img_grid = make_grid(fake, normalize=True, value_range=(-1, 1))
        writer.add_image("WGAN/Generated Images", img_grid, global_step=epoch)

    print(f"[Epoch {epoch+1}/{epochs}] Critic Loss: {loss_critic.item():.4f} | Gen Loss: {loss_generator.item():.4f}")
