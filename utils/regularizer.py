import torch


def compute_deformation_jacobian(
        points: torch.Tensor, transformed_points: torch.Tensor
) -> torch.Tensor:
    '''
    points shape: ..., n
    transformed_points shape: ..., m
    output shape: ..., m, n
    '''
    deriv_list = []
    output_spatial_dims = transformed_points.shape[-1]
    for dim in range(output_spatial_dims):
        deriv = torch.autograd.grad(
            torch.sum(transformed_points[..., dim]),
            points, retain_graph=True, create_graph=True
        )[0][..., None]
        deriv_list.append(deriv)
    return torch.cat(deriv_list, dim=-1)
