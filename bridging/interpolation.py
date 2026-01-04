import torch

def linear_interpolate(v0, v1, t):
    """
    Standard Linear Interpolation (LERP).
    Formula: v0 + t * (v1 - v0)
    
    Args:
        v0 (torch.Tensor): Starting latent tensor (B, C, W)
        v1 (torch.Tensor): Ending latent tensor (B, C, W)
        t (float): Interpolation factor [0.0, 1.0]
        
    Returns:
        torch.Tensor: Interpolated tensor
    """
    return v0 + t * (v1 - v0)


def slerp(v0, v1, t, DOT_THRESHOLD=0.9995):
    """
    Spherical Linear Interpolation (SLERP) - Global.
    Treats the whole (C, W) block as one vector.
    """
    # Ensure inputs are tensors
    if not isinstance(v0, torch.Tensor): v0 = torch.tensor(v0)
    if not isinstance(v1, torch.Tensor): v1 = torch.tensor(v1)
    
    # Copy to avoid modifying originals
    v0_copy = v0.clone()
    v1_copy = v1.clone()
    
    batch_size, channels, width = v0_copy.shape
    
    # Flatten the feature dimensions (C, W) to treat them as vectors
    # Shape becomes (B, C*W)
    v0_flat = v0_copy.view(batch_size, -1)
    v1_flat = v1_copy.view(batch_size, -1)
    
    # Calculate magnitudes (norms)
    # v0_norm: (B, 1)
    v0_norm = torch.norm(v0_flat, dim=1, keepdim=True)
    v1_norm = torch.norm(v1_flat, dim=1, keepdim=True)
    
    # Normalize vectors to unit sphere for angle calculation
    v0_normed = v0_flat / (v0_norm + 1e-8)
    v1_normed = v1_flat / (v1_norm + 1e-8)
    
    # Calculate dot product
    # dot: (B, 1)
    dot = torch.sum(v0_normed * v1_normed, dim=1, keepdim=True)
    
    # Clamp dot product to [-1, 1] to avoid NaN in arccos
    dot = torch.clamp(dot, -1.0, 1.0)
    
    # --- Handling Parallel Vectors (Collinear) ---
    is_close = dot > DOT_THRESHOLD
    
    # Calculate Omega (angle)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    
    # Calculate Coefficients
    c0 = torch.sin((1.0 - t) * omega) / sin_omega
    c1 = torch.sin(t * omega) / sin_omega
    
    # Interpolate on unit sphere
    res_unit = c0 * v0_normed + c1 * v1_normed
    
    # Linearly interpolate magnitude
    mag_res = (1.0 - t) * v0_norm + t * v1_norm
    
    # Combine
    res_flat = res_unit * mag_res
    
    # --- Apply LERP Fallback for collinear vectors ---
    if torch.any(is_close):
        lerp_flat = linear_interpolate(v0_flat, v1_flat, t)
        mask = is_close.expand_as(res_flat)
        res_flat = torch.where(mask, lerp_flat, res_flat)
        
    # Reshape back to (B, C, W)
    return res_flat.view(batch_size, channels, width)

def slerp_channel_wise(v0, v1, t):
    """
    Channel-wise SLERP.
    Performs SLERP independently for each channel.
    This respects the independent geometry of each feature channel.
    
    Args:
        v0 (torch.Tensor): (B, C, W)
        v1 (torch.Tensor): (B, C, W)
        t (float): Interpolation factor
    """
    b, c, w = v0.shape
    
    # Reshape to treat each channel as an independent sample in a larger batch
    # New shape: (B*C, 1, W)
    # This tricks the global slerp function into calculating B*C different angles
    v0_reshaped = v0.view(b * c, 1, w)
    v1_reshaped = v1.view(b * c, 1, w)
    
    # Apply standard SLERP logic on the expanded batch
    res_reshaped = slerp(v0_reshaped, v1_reshaped, t)
    
    # Reshape back to original dimensions
    return res_reshaped.view(b, c, w)


def bezier_2nd_order(v0, v1, v2, t):
    """
    2nd Order Bezier Interpolation.
    Formula: (1-t)^2 * v0 + 2(1-t)t * v1 + t^2 * v2
    
    Args:
        v0 (torch.Tensor): Starting latent tensor (B, C, W)
        v1 (torch.Tensor): Control point latent tensor (B, C, W)
        v2 (torch.Tensor): Ending latent tensor (B, C, W)
        t (float): Interpolation factor [0.0, 1.0]
        
    Returns:
        torch.Tensor: Interpolated tensor
    """
    return ((1 - t) ** 2) * v0 + 2 * (1 - t) * t * v1 + (t ** 2) * v2