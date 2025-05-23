import torch

def stable_inversion(M):
    L = torch.linalg.cholesky_ex(M + torch.eye(M.shape[-1])*0.00001)[0]
    return torch.cholesky_inverse(L)

def batched_block_diag(M):
    """Generalises torch.block_diag to tensors of shape (batch, a, b, b).
    
    Naive implementation would be 
        torch.cat(*[torch.block_diag(M[i]).unsqueeze(0) for i in range(M.shape[0])], dim=0)

    Here we implement it in a vectorised way.
    """
    batch, a, b, _  = M.shape

    out = torch.zeros((batch, a*b, a*b))
    
    pass