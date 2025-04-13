lambda_gp = 10 
n_critic = 5    
epochs = 50

for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        real_imgs = imgs.to(device)


        #  Train Discriminator
        optimizer_D.zero_grad()

        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = generator(z).detach()
        
        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(fake_imgs)
        gp = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gp

        d_loss.backward()
        optimizer_D.step()

        # Train Generator every n_critic steps
        if i % n_critic == 0:
            optimizer_G.zero_grad()

            z = torch.randn(imgs.size(0), latent_dim).to(device)
            gen_imgs = generator(z)
            g_loss = -torch.mean(discriminator(gen_imgs))

            g_loss.backward()
            optimizer_G.step()

            # TensorBoard logging
            global_step = epoch * len(train_loader) + i
            writer.add_scalar("Loss/WGAN-GP_Discriminator", d_loss.item(), global_step)
            writer.add_scalar("Loss/WGAN-GP_Generator", g_loss.item(), global_step)

    print(f"[Epoch {epoch+1}/{epochs}] D loss: {d_loss.item():.4f} | G loss: {g_loss.item():.4f}")

    if epoch % 10 == 0:
        with torch.no_grad():
            z = torch.randn(25, latent_dim).to(device)
            sample_imgs = generator(z)
            grid = torchvision.utils.make_grid(sample_imgs, nrow=5, normalize=True)
            writer.add_image("WGAN-GP Generated Samples", grid, epoch)
