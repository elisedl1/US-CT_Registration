import torch

class HessianSimilarity:
    def __init__(
        self,
        reference_hessians: torch.Tensor,
        reference_gradients: torch.Tensor,
        template_hessians: torch.Tensor
    ) -> None:
        self.ref_hess = self.vectorize_matrices(reference_hessians)
        self.ref_dyadic = self.vectorize_matrices(
            self.dyadic_product(*2*(reference_gradients,))
        )
        self.tem_hess = self.vectorize_matrices(template_hessians)

    def compute_map(self) -> torch.Tensor:
        measure = CoplanarityMeasure(
            self.ref_hess, self.ref_dyadic, self.tem_hess
        )
        return measure.compute()

    @staticmethod
    def dyadic_product(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        '''
        computes dyadic product of u and v
        '''
        u = u.unsqueeze(-1)
        v = v.unsqueeze(-1)
        return u @ v.transpose(-1, -2)

    @staticmethod
    def vectorize_matrices(matrices: torch.Tensor) -> torch.Tensor:
        shape = matrices.shape
        vector_shape = shape[:-2] + (shape[-2]*shape[-1],)
        return matrices.reshape(vector_shape)


class CoplanarityMeasure:
    def __init__(
            self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ) -> None:
        '''
        Computes the coplanarity of inputs in the form of:
        z = alpha * x + beta * y
        where x, y and z are considered to be tensors.
        '''
        self.xx = self.vector_dot(x, x)
        self.yy = self.vector_dot(y, y)
        self.zz = self.vector_dot(z, z)
        self.xy = self.vector_dot(x, y)
        self.xz = self.vector_dot(x, z)
        self.yz = self.vector_dot(y, z)
        self.determinant = self.xx*self.yy - self.xy**2

    def compute(self) -> torch.Tensor:
        numerator = self.xx * self.yz**2
        numerator += self.yy * self.xz**2
        numerator -= 2 * self.xy * self.xz * self.yz
        return numerator / self.determinant

    @staticmethod
    def vector_dot(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        return torch.sum(vector1 * vector2, axis=-1)


class GradientSimilarity:
    def __init__(
        self,
        reference_gradients: torch.Tensor,
        template_gradients: torch.Tensor
    ) -> None:
        self.reference_gradients = reference_gradients
        self.template_gradients = template_gradients

    def compute_map(self) -> torch.Tensor:
        u_mag_sq = self.vector_dot(* 2 * (self.reference_gradients,))
        v_mag_sq = self.vector_dot(* 2 * (self.template_gradients,))
        v_dot_u = self.vector_dot(
            self.reference_gradients, self.template_gradients
        )
        return v_dot_u ** 2 / (u_mag_sq * v_mag_sq)

    @staticmethod
    def vector_dot(vector1: torch.Tensor, vector2: torch.Tensor) -> torch.Tensor:
        return torch.sum(vector1 * vector2, axis=-1)


class LC2:
    def __init__(
        self,
        reference_gradients: torch.Tensor,
        reference_hg: torch.Tensor,
        template_gradients: torch.Tensor
    ) -> None:
        self.ref_grad = reference_gradients
        self.ref_hg = reference_hg
        self.tem_grad = template_gradients

    def compute_map(self) -> torch.Tensor:
        measure = CoplanarityMeasure(
            self.ref_grad, self.ref_hg, self.tem_grad
        )
        return measure.compute()


class IntensitySimilarity:
    """
    Computes simple intensity similarity (mean product) between two tensors.
    """

    @staticmethod
    def compute(fixed_intensities: torch.Tensor, moving_intensities: torch.Tensor) -> float:

        product_intensities = fixed_intensities * moving_intensities
        return product_intensities.mean().item()

    
