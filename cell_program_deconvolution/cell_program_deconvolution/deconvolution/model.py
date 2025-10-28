import torch
import torch.nn as nn

class DeconvModel(nn.Module):
    def __init__(self, T, P, S, X_tensor, L):
        super().__init__()
        self.X = X_tensor  # shape: (S, T)
        self.L = L         # shape: (S, S)
        self.Y_tps = nn.Parameter(torch.rand(T, P, S))  # learnable

    def forward(self):
        Y_pred = torch.einsum('st,tps->ps', self.X, self.Y_tps)
        return Y_pred

    def loss(self, Y_obs, lambda1=1e-4, lambda2=1e-2):
        Y_pred = self.forward()
        recon_loss = nn.functional.mse_loss(Y_pred, Y_obs)
        l1_loss = self.Y_tps.abs().sum()

        smooth_loss = 0.
        for t in range(self.Y_tps.shape[0]):
            for p in range(self.Y_tps.shape[1]):
                y = self.Y_tps[t, p, :].unsqueeze(0)
                val = torch.matmul(torch.matmul(y, self.L), y.t()).squeeze()
                smooth_loss += val
        
        total_loss = recon_loss + lambda1 * l1_loss + lambda2 * smooth_loss

        return total_loss, recon_loss, l1_loss, smooth_loss