import dataclasses

import numpy as np
import scipy

@dataclasses.dataclass
class ParametersGaussianPlume:
    grid_size: int      # length of each dimension
    T_max: int          # maximum episode length
    A0: float           # release rate, eq(11) on page 6
    u: float            # wind speed
    phi: float          # wind direction
    d: float            # diffusion coefficient
    tau: float          # particle lifetime
    p_d: float          # detection probability
    sigma_k: float      # background noise level

    def __post_init__(self):
        assert self.grid_size > 0
        assert self.T_max > 0
        assert self.A0 > 0
        assert self.u >= 0
        assert self.phi >= 0 and self.phi < 2 * np.pi
        assert self.d > 0
        assert self.tau > 0
        assert self.p_d > 0 and self.p_d <= 1
        assert self.sigma_k > 0

        self.lambda_sensor = np.sqrt((self.d * self.tau)/(1 + (self.tau*self.u**2)/(4*self.d))) # eq(9) on page 5
        self.scale_parameter = 3 / self.grid_size # According to Table 1 on page 9


@dataclasses.dataclass
class ParametersBase:
    grid_size: int  # length of each dimension
    h_max: int  # maximum number of hits where h in [0, 1, ..., h_max]
    T_max: int  # maximum episode length
    lambda_over_delta_x: float  # dispersion length-scale of particles in medium (lambda) / cell size (delta_x)
    delta_x_over_a: float  # cell size (delta_x) / agent radius (a)

    def __post_init__(self):
        assert self.grid_size > 0
        assert self.h_max > 0
        assert self.T_max > 0
        assert self.lambda_over_delta_x > 0
        assert self.delta_x_over_a > 0

        self.h = np.arange(0, self.h_max + 1)


@dataclasses.dataclass
class ParametersIsotropic(ParametersBase):
    R_times_delta_t: float  # source emission rate (R) * sniff time (delta_t)

    def __post_init__(self):
        super().__post_init__()
        assert self.R_times_delta_t > 0

        self.lambda_over_a = self.lambda_over_delta_x * self.delta_x_over_a
        self.mu0_Poisson = (
            1 / np.log(self.lambda_over_a) * scipy.special.k0(1)
        ) * self.R_times_delta_t


@dataclasses.dataclass
class ParametersWindy(ParametersBase):
    R_bar: float  # 0.5 * R_times_delta_t
    V_bar: float  # mean wind speed (V) * cell size (delta_x) / effective diffusivity (D)
    tau_bar: float  # mean wind speed squre (V^2) * particle lifetime (tau) / effective diffusivity (D)

    def __post_init__(self):
        super().__post_init__()
        assert self.R_bar > 0
        assert self.V_bar > 0
        assert self.tau_bar > 0

        self.V_over_2D = 0
        self.lambda_over_delta_x = np.sqrt(
            (self.tau_bar / self.V_bar**2) / (1 + self.tau_bar / 4)
        )
        self.lambda_over_a = self.lambda_over_delta_x * self.delta_x_over_a


SMALLER_ISOTROPIC_DOMAIN = ParametersIsotropic(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    R_times_delta_t=1.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)

LARGER_ISOTROPIC_DOMAIN = ParametersIsotropic(
    grid_size=53,
    h_max=3,
    T_max=2188,
    lambda_over_delta_x=3.0,
    R_times_delta_t=2.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
)

SMALLER_WINDY_DOMAIN_WITH_DETECTION = ParametersWindy(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
    R_bar=2.5,  # with detections
    V_bar=2.0,
    tau_bar=150.0,
)

SMALLER_WINDY_DOMAIN_WITHOUT_DETECTION = ParametersWindy(
    grid_size=19,
    h_max=2,
    T_max=642,
    lambda_over_delta_x=1.0,
    delta_x_over_a=2.0,  # missing from paper, hard-coded in implementation
    R_bar=0.25,  # almost without detections
    V_bar=2.0,
    tau_bar=150.0,
)

SMALLER_GAUSSIAN = ParametersGaussianPlume(
    grid_size=19,
    T_max=642,
    A0=0.75,
    u=1.0,
    phi=0.25*np.pi,
    d=7.0,
    tau=0.7,
    p_d=0.7,
    sigma_k=0.05
)