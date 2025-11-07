import torch

def train_model(model, Y_obs, num_epochs=3000, lr=1e-3, lambda1=1e-4, lambda2=1e-2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for epoch in range(num_epochs):
        loss, recon = model.loss(Y_obs, lambda1, lambda2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss={loss.item():.4f}, Recon={recon.item():.4f}")
        history.append((loss.item(), recon.item()))

    return history
