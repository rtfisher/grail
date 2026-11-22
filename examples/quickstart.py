"""A self-contained matrix-free response with known amplification four."""
import torch
from grail import resolvent_solve


def main():
    x = torch.zeros(2, dtype=torch.float64)
    residual = torch.tensor([.01, -.02], dtype=torch.float64)
    result = resolvent_solve(x, residual, emulator_jvp=lambda v: .75*v,
                             host_jvp=lambda v: v, gmres_tol=1e-12)
    torch.testing.assert_close(result.delta_x_star, 4*residual)
    print(result.delta_x_star)


if __name__ == "__main__":
    main()
