import torch
import torch.nn.functional as F

def random_affine_transform(z,uv, degrees, translate, scale, is_train = True):

    _, C, H, W = uv.shape  # uv is [1, C, H, W]
    B,_ = z.shape

    # Generate affine transformation parameters
    if is_train:
        angles = torch.empty(B, device=z.device).uniform_(*degrees)
        translations = (
            torch.empty(B, 2, device=z.device).uniform_(-1, 1)
            * torch.tensor(translate, device=z.device)
            * torch.tensor([W, H], device=z.device)
        )
        scales = torch.empty(B, device=z.device).uniform_(*scale)
    else:
        angles = torch.full((B,), sum(degrees) / 2, device=z.device)
        translations = torch.tensor(translate, device=z.device).expand(B, 2)
        scales = torch.full((B,), sum(scale) / 2, device=z.device)

    # Compute affine transformation matrices
    angles_rad = torch.deg2rad(angles)
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)

    affine_matrices = torch.zeros((B, 3, 3), device=z.device)
    affine_matrices[:, 0, 0] = cos_a * scales
    affine_matrices[:, 0, 1] = -sin_a * scales
    affine_matrices[:, 1, 0] = sin_a * scales
    affine_matrices[:, 1, 1] = cos_a * scales
    affine_matrices[:, 0, 2] = translations[:, 0]
    affine_matrices[:, 1, 2] = translations[:, 1]
    affine_matrices[:, 2, 2] = 1.0  # Homogeneous coordinate

    # Prepare UV for affine transformation
    uv_permuted = uv.permute(0, 2, 3, 1)  # [B, H, W, C]
    ones = torch.ones_like(uv_permuted[..., :1])  # Add homogeneous coordinate
    uv_homogeneous = torch.cat([uv_permuted, ones], dim=-1)  # [B, H, W, C+1]

    # Apply affine transformation to UV
    uv_transformed = torch.einsum("bhwc,bca->bhwa", uv_homogeneous, affine_matrices)  # [B, H, W, C+1]
    uv_transformed = uv_transformed[..., :-1]  # Remove homogeneous coordinate
    uv_transformed = uv_transformed.permute(0, 3, 1, 2)  # Restore to [B, C, H, W]

    # Process latent variables
    z_expanded = z[:, :, None, None].expand(B, -1, H, W)  # [B, latent_dim, H, W]

    # Combine latent variables with transformed UV
    outputs = torch.cat([z_expanded, uv_transformed], dim=1)  # [B, latent_dim + C, H, W]

    return outputs


# Example usage
if __name__ == "__main__":
    # Create a batch of random images (B=4, C=3, H=128, W=128)
    B, C, H, W = 4, 3, 128, 128
    images = torch.randn(B, C, H, W)

    # Define random parameters for transformations per batch
    degrees = (-30, 30)  # Rotation range
    translate = (0.5, 0.5)  # Translation as a fraction of image size
    scale = (0.8, 1.2)  # Scaling range

    # Apply random affine transformations
    transformed_images = random_affine_transform(z,images, degrees, translate, scale)

    print("Original shape:", images.shape)
    print("Transformed shape:", transformed_images.shape)
