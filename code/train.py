# Training loop
for epoch in range(start_epoch, epochs):
    for i, (L, ab) in enumerate(dataloader):
        L, ab = L.to(device), ab.to(device)

        optimizer.zero_grad()
        L = L.float()

        # Print the shape of L and ab before the forward pass
        print(f"Input L shape: {L.shape}")  # Expected: [batch_size, 1, H, W]
        print(f"Target ab shape: {ab.shape}")  # Expected: [batch_size, 2, H, W]

        # Model forward pass
        output = model(L)

        # Print the shape of model output
        print(f"Model output shape: {output.shape}")  # Expected: [batch_size, 2, H, W]

        # Resize ab channels to match the output size
        ab_resized = F.interpolate(ab, size=(output.shape[2], output.shape[3]), mode='bilinear', align_corners=False)

        # Print the shape of ab_resized
        print(f"Resized ab shape: {ab_resized.shape}")  # Expected: [batch_size, 2, H, W]

        # Correcting the channel dimension order by permuting ab_resized
        ab_resized = ab_resized.permute(0, 3, 1, 2)  # Reorder to (batch_size, 2, H, W)

        # Loss calculation
        loss = criterion(output, ab_resized)
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.8f}')

    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, 'checkpoint.pth')
    print("Checkpoint saved!")
