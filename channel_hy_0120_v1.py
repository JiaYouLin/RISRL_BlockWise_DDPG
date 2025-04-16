import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from environment import Environment
import einops
from utils import scenario_configs

import os
from math import sqrt, pi, log10
from typing import List, Tuple

"""
Settings
"""
os.chdir(os.path.dirname(os.path.abspath(__file__)))

"""
Main Classes
"""


class Channel(Environment):
    def __init__(
        self,
        wavelength: float,
        d_ris_elem: float,
        ris_center: Tuple[float],
        ris_norm: Tuple[float],
        ris_size: Tuple[int],
        area: Tuple[float],
        BS_pos: Tuple[float],
        M: int,
        K: int,
        device: str,
        MU_dist: str,
        rand_seed: int = 1,
    ):
        """The object describing the channels in the simulation environment.

        Args:
            wavelength: The radio wavelength used by the base station (BS).
            d_ris_elem: RIS reflective element spacing (m)
            ris_center: The position of the center of RIS. (x, y, z)
            ris_norm: The normal vector of RIS.
            ris_size: The size of RIS. (height, width)
            area: The size of simulation area, (L_x, L_y). Note that the origin (0, 0) locates at the center of the simulation area.
            BS_pos: The position of the base station (BS). (x, y, z)
            M: Number of antennas at BS (Tx).
            K: Number of users (Rx).
            device: 'cpu' or 'cuda:<num>'
            MU_dist: The distribution rule of users. 'poisson', 'normal'
        """
        # Initialize
        super().__init__(wavelength, d_ris_elem, ris_center, ris_norm, ris_size, area, BS_pos, M, K, device, MU_dist, rand_seed)
        super().__init_environment__()
        self.__init_channel__()
        self.to(device)

    def to(self, device: str):
        """Move tensors onto the specific device.

        Args:
            device: 'cpu' or 'cuda:<num>'

        Returns:
            The object on the new device
        """
        super().to(device)
        self.device = device
        return self

    def __init_channel__(self):
        """Update the channel only

        Returns:
            The new object.
        """
        self.u_Bk, self.u_BS, self.u_Sk = None, None, None
        self.H_BS, self.h_Sk, self.h_Bk = None, None, None  # (N, 1), (K, N), (K, 1)
        self.AWGN_norm = None
        return self

    def update_environment(self) -> None:
        """Update the environment and channel

        Returns:
            The new object.
        """
        super().update_environment()
        self.__init_channel__()
        return self

    def __generate_channels__(
        self,
        eta: float,
        zeta: float,
        alpha_los: float,
        alpha_nlos: float,
    ) -> None:
        """
        Args:
            eta, zeta: coefficients about Rician K-factor
            alpha_los, alpha_nlos: propagation decay factor (for power)

        Returns:
            This function creates the following matrices:
            * h_Bk (K, M)
            * h_Sk (K, N)
            * H_BS (N, M)
        """
        if (self.H_BS is None) or (self.h_Sk is None) or (self.h_Bk is None):
            self.getBlockMats()
            self.getDistMats()
            self.__generate_channel_stochastic_components__()

            # BS-MU: (K, M)
            # LoS: eta是LoS比例，self.B_Bk是阻塞矩陣，代表BS和UE是否有直接視距連線，torch.polar()用於計算複數表示的相位移，基於距離D_Bk和波長wavelength
            # NLoS: zeta控制NLoS比例，self.u_Bk是模擬隨機小尺度衰落的變數，D_Bk.pow(-alpha_nlos / 2)是基於NLoS路徑損耗指數的距離衰減
            # self.h_Bk = C0 * (
            #     eta * self.B_Bk.type(torch.cfloat) * torch.polar(self.D_Bk.pow(-alpha_los / 2), -2 * pi * self.D_Bk / wavelength)  # LoS component
            #     + (1 - (1 - zeta) * self.B_Bk.type(torch.cfloat)) * self.D_Bk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Bk  # NLoS component
            # )
            # # RIS-MU: (K, N)
            # self.h_Sk = C0 * (
            #     eta * self.B_Sk.type(torch.cfloat) * torch.polar(self.D_Sk.pow(-alpha_los / 2), -2 * pi * self.D_Sk / wavelength)  # LoS component
            #     + (1 - (1 - zeta) * self.B_Sk.type(torch.cfloat)) * self.D_Sk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Sk  # NLoS component
            # )
            # # BS-RIS: (N, M)
            # self.H_BS = C0 * (
            #     eta * self.B_BS.type(torch.cfloat) * torch.polar(self.D_BS.pow(-alpha_los / 2), -2 * pi * self.D_BS / wavelength)  # LoS component
            #     + (1 - (1 - zeta) * self.B_BS.type(torch.cfloat)) * self.D_BS.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_BS  # NLoS component
            # )
            M, K, N = self.M, self.K, self.N
            q = 1
            gamma = 2 * (2 * q + 1)

            # D_BS = torch.mean(self.D_BS, dim=1)       # (N)
            # D_Sk = self.D_Sk                          # (K, N)
            # D = torch.squeeze(D_BS.unsqueeze(dim=0) + D_Sk)          # (K, N)
            # phase = 2*pi*(D)/wavelength
            # Theta = torch.diag(
            #     torch.polar(
            #         torch.ones_like(D, dtype=torch.float, device=device),
            #         phase,
            #     )
            # )
            # print(torch.mean(D), alpha_los, alpha_nlos)
            # Reference:
            # Pathloss model: Chu, Z., Xiao, P., Mi, D., Chen, H., & Hao, W. (2020). Intelligent reflecting surfaces enabled cognitive internet of things based on practical pathloss model. China Communications, 17(12), 1-16.
            # Rician model: Xie, Z., Yi, W., Wu, X., Liu, Y., & Nallanathan, A. (2022). Downlink multi-RIS aided transmission in backhaul limited networks. IEEE Wireless Communications Letters, 11(7), 1458-1462.

            # BS-MU: (K, M)
            # self.h_Bk = (wavelength)/(4*pi) * (
            #     self.B_Bk.type(torch.cfloat) * self.D_Bk.pow(-alpha_los / 2).type(torch.cfloat) * (eta*torch.polar(torch.ones_like(self.D_Bk,device=device), -2 * pi * self.D_Bk / wavelength) + zeta*self.u_Bk)# LoS component
            #     + (1-self.B_Bk.type(torch.cfloat)) * self.D_Bk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Bk  # NLoS component
            # )
            # RIS-MU: (K, N)
            # self.h_Sk =  (
            #     self.B_Sk.type(torch.cfloat) * self.D_Sk.pow(-alpha_los / 2).type(torch.cfloat) * (eta*torch.polar(torch.ones_like(self.D_Sk,device=device), -2 * pi * self.D_Sk / wavelength) +  zeta*self.u_Sk)# LoS component
            #     + (1-self.B_Sk.type(torch.cfloat)) * self.D_Sk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Sk  # NLoS component
            # )
            # BS-RIS: (N, M)
            # print("-1-1-1-1", (torch.abs(self.cos_angle_in) ))
            # self.H_BS = d_ant**2/(4*pi) * torch.abs(self.cos_angle_in) * (
            #     self.B_BS.type(torch.cfloat) * self.D_BS.pow(-alpha_los / 2).type(torch.cfloat) * (eta*torch.polar(torch.ones_like(self.D_BS,device=device), -2 * pi * self.D_BS / wavelength) + zeta*self.u_BS) # LoS component
            #     + (1-self.B_BS.type(torch.cfloat)) * self.D_BS.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_BS  # NLoS component
            # )
            # a1 = self.h_Sk
            # b1 = self.H_BS
            # x1 = self.h_Sk@self.H_BS
            # print(x1)

            # Reference: Feng, C., Lu, H., Zeng, Y., Li, T., Jin, S., & Zhang, R. (2023). Near-field modelling and performance analysis for extremely large-scale IRS communications. IEEE Transactions on Wireless Communications.
            # Pathloss
            # BS-MU
            G_Bk = self.wavelength / (4 * pi)  # (K, M)
            self.h_Bk = G_Bk * (
                self.B_Bk.type(torch.cfloat)
                * self.D_Bk.pow(-alpha_los / 2).type(torch.cfloat)
                * (
                    eta * torch.polar(torch.ones_like(self.D_Bk, device=self.device), -2 * pi * self.D_Bk / self.wavelength) + zeta * self.u_Bk
                )  # LoS component
                + (1 - self.B_Bk.type(torch.cfloat)) * self.D_Bk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Bk  # NLoS component
            )

            # RIS-MU
            # print("0000", ( (torch.maximum(self.Phi_Sk, torch.zeros(K, N, device=device))/(self.D_Sk/self.Dc_Sk))), self.Phi_Sk, self.D_Sk/self.Dc_Sk)
            G_Sk = (
                (self.wavelength / (4 * pi))
                * sqrt(gamma)
                * (torch.maximum(self.Phi_Sk, torch.zeros(K, N, device=self.device)) / (self.D_Sk / self.Dc_Sk)) ** q
            )  # (K, N)
            self.h_Sk = G_Sk * (
                self.B_Sk.type(torch.cfloat)
                * self.D_Sk.pow(-alpha_los / 2).type(torch.cfloat)
                * (
                    eta * torch.polar(torch.ones_like(self.D_Sk, device=self.device), -2 * pi * self.D_Sk / self.wavelength) + zeta * self.u_Sk
                )  # LoS component
                + (1 - self.B_Sk.type(torch.cfloat)) * self.D_Sk.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_Sk  # NLoS component
            )

            # BS-RIS
            # print("1111", ((torch.maximum(self.Phi_BcS, torch.zeros(N, M, device=device))/(self.D_BS/self.Dc_BcS))), self.Phi_BcS)
            G_BS = (
                (self.wavelength / (4 * pi))
                * sqrt(gamma)
                * (torch.maximum(self.Phi_BSc, torch.zeros(N, M, device=self.device)) / (self.D_BS / self.Dc_BcSc)) ** q
            )  # (N, M)
            self.H_BS = G_BS * (
                self.B_BS.type(torch.cfloat)
                * self.D_BS.pow(-alpha_los / 2).type(torch.cfloat)
                * (
                    eta * torch.polar(torch.ones_like(self.D_BS, device=self.device), -2 * pi * self.D_BS / self.wavelength) + zeta * self.u_BS
                )  # LoS component
                + (1 - self.B_BS.type(torch.cfloat)) * self.D_BS.pow(-alpha_nlos / 2).type(torch.cfloat) * self.u_BS  # NLoS component
            )
            print(torch.abs(torch.sum(self.h_Bk, 1)), torch.abs(torch.sum(self.H_BS)), torch.abs(torch.sum(self.h_Sk, 1)))
            # a2 = self.h_Sk
            # b2 = self.H_BS
            # X1 =  self.h_Bk
            # X2 = self.h_Sk@self.H_BS
            # print(torch.mean(torch.abs(X1)**2), torch.mean(abs(X2)**2))
            # print(x2)
            # print(a1/a2)
            # print(b1/b2)
            # print(x1/x2)

            # print(G_Bk, G_Sk, G_BS)
            # exit()

    def __generate_channel_stochastic_components__(self, BS_Tx_power=16, noise_floor=-90):
        """Compute the stochastic components of the channels

        Args:
            BS_Tx_power (int, optional): The Tx power of BS in dBm. Defaults to 16.
            noise_floor (int, optional): The noise floor of the environemnt in dBm. Defaults to -90.
        """
        if (self.u_Bk is None) or (self.u_Sk is None) or (self.u_BS is None):
            M, N, K = self.M, self.N, self.K  # K表示UE數量, M表示BS天線數量

            self.u_Bk = torch.view_as_complex(
                torch.normal(torch.zeros((K, M, 2), device=self.device), 1 / sqrt(2), generator=self.generator)
            )  # 根據正態分佈生成的隨機數組成的複數張量, BS和UE間的隨機NLoS成分
            self.u_Sk = torch.view_as_complex(
                torch.normal(torch.zeros((K, N, 2), device=self.device), 1 / sqrt(2), generator=self.generator)
            )  # RIS和UE間的隨機NLoS成分
            self.u_BS = torch.view_as_complex(
                torch.normal(torch.zeros(size=(N, M, 2), device=self.device), 1 / sqrt(2), generator=self.generator)
            )  # BS和RIS間的隨機NLoS成分

            mean_SNR = BS_Tx_power - noise_floor - 10 * log10(M * K)
            self.mean_SNR = 10 ** (mean_SNR / 10)
            self.AWGN_norm = torch.view_as_complex(
                torch.normal(torch.zeros(size=(K, 1, 2), device=self.device), 1 / sqrt(2), generator=self.generator)
            ) / sqrt(10 ** (mean_SNR / 10))

    def update_channel(
        self,
        alpha_los: float = 2,  # 2,
        alpha_nlos: float = 4,  # orig: 2.5, now: 4
        kapa: float = 10,  # 10 dB
        time_corrcoef: float = 0,
    ) -> None:
        self.getBlockMats()
        self.getDistMats()

        eta, zeta = sqrt(kapa / (1 + kapa)), sqrt(1 / (1 + kapa))
        is_time_correlated_channel = time_corrcoef != 0 and (self.H_BS is not None) and (self.h_Sk is not None) and (self.h_Bk is not None)
        if is_time_correlated_channel:
            assert time_corrcoef < 1 and time_corrcoef >= 0
            old_h_Bk, old_H_BS, old_h_Sk, old_AWGN_norm = self.h_Bk, self.H_BS, self.h_Sk, self.AWGN_norm

        self.__init_channel__()
        self.__generate_channels__(eta, zeta, alpha_los, alpha_nlos)
        if is_time_correlated_channel:
            self.h_Bk = time_corrcoef * old_h_Bk + sqrt(1 - time_corrcoef**2) * self.h_Bk
            self.H_BS = time_corrcoef * old_H_BS + sqrt(1 - time_corrcoef**2) * self.H_BS
            self.h_Sk = time_corrcoef * old_h_Sk + sqrt(1 - time_corrcoef**2) * self.h_Sk
            self.AWGN_norm = time_corrcoef * old_AWGN_norm + sqrt(1 - time_corrcoef**2) * self.AWGN_norm

    def get_channel_coefficient(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get h_Bk, h_BS, h_Sk, and (noise_density/per_UE_Pt_density)

        Returns:
            h_Bk (torch.Tensor): BS-UE, (K, M)
            h_BS (torch.Tensor): BS-RIS (N, M)
            h_Sk (torch.Tensor): RIS-UE (K, N)
            AWGN_norm (torch.Tensor): h_Bk (1)
        """
        return self.h_Bk, self.H_BS, self.h_Sk, self.AWGN_norm

    def get_joint_channel_coefficient(
        self,
        Z_Theta: torch.Tensor,
        batch_size: int = None,
        progress: bool = False,
    ) -> torch.Tensor:
        """Get joint channel

        Args:
            Z_Theta (torch.Tensor): Z*Theta, (Z, N)
            batch_size (int, optional): _description_. Defaults to None (not batched).
            progress (bool, optional): Show progress bar. Defaults to False.

        Returns:
            H (torch.Tensor): Joint channel coefficients, (Z, K, M)
        """
        if (self.H_BS is None) or (self.h_Sk is None) or (self.h_Bk is None):
            print("Run Channel.update_channel() beforehand")
            exit()

        device = self.device
        Z_Theta = Z_Theta.to(device)
        K, M = self.K, self.M

        H = torch.zeros((len(Z_Theta), K, M), dtype=torch.cfloat, device=device)
        count = 0
        if progress:  # show progressbar
            pbar = tqdm.tqdm(range(len(Z_Theta)))

        # parallelism
        if batch_size is None:
            batch = len(Z_Theta)
        else:
            batch = min(batch_size, len(Z_Theta))

        while count < len(Z_Theta):
            start = count
            end = min(len(Z_Theta), count + batch)
            # dec ompose to increase gpu memory efficiency of einsum:
            tempMat = torch.einsum("zn,nm->znm", Z_Theta[start:end], self.H_BS)  # (Z, N, M)
            H[start:end] = self.h_Bk + torch.einsum("kn,znm->zkm", self.h_Sk, tempMat)  # (Z, K, M)
            if progress:  # show progressbar
                pbar.update(n=min(batch, len(Z_Theta) - count))
            count += batch
        return H  # (Z, K, M), (K)

    def SINR(self, H: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        """Return the SINR matrix. (Z, K)

        Args:
            H: channel coefficient with shape.  (Z, K, M)
            W: beamformer with shape.           (Z, K, M)

        Returns:
            torch.Tensor: the SINR of each user. (Z, K)
        """
        device = self.device
        H = H.to(device)

        HW = torch.abs(H @ W.transpose(-1, -2))
        assert len(HW.shape) == 3
        assert HW.shape[-1] == HW.shape[-2]

        S = HW.diagonal(dim1=-2, dim2=-1)
        I = HW.sum(dim=-1) - S

        P_S = torch.abs(S) ** 2
        P_I = torch.abs(I) ** 2
        P_N = torch.abs(self.AWGN_norm) ** 2
        SINR = P_S / (P_I + P_N.view(1, -1))

        # return self.sinr_linear  # SINR: (Z, K)
        return SINR  # SINR: (Z, K)

    def plot_SINR(self, dir: str, SINR: torch.Tensor) -> None:
        """Plot the SINR of MU from the top view.

        Args:
            dir: The directory to dump the figure.
            SINR: SINR of each user. (K, 1)
            batch_size: Used to prevent CUDA out of memory. Decrease this value if CUDA out of memory happens, or a larger value computes faster.
        """
        BS_pos = self.BS_pos
        fig, ax = plt.subplots()

        # RIS
        risproj1, _ = torch.min(self.ris_pos.view(-1, 3)[:, :2], 0)
        risproj2, _ = torch.max(self.ris_pos.view(-1, 3)[:, :2], 0)
        p = torch.cat([risproj1.unsqueeze(0), risproj2.unsqueeze(0)], dim=0)
        if (risproj1 - risproj2)[0] * (risproj1 - risproj2)[0] < 0:
            p = torch.cat([risproj1.unsqueeze(0), risproj2.flip([0]).unsqueeze(0)], dim=0)
        p = p.cpu().numpy()
        ax.plot(p[:, 0], p[:, 1], c="royalblue", label="RIS", linewidth=1)

        for obs in self.obstacles:  # obstacles
            base = torch.cat([obs.proj2D(), obs.proj2D()[0].unsqueeze(0)], dim=0).cpu().numpy()
            ax.plot(base[:, 0], base[:, 1], c="darkorange", label="Obstacles", linewidth=1)

        # SINR (K, 1)
        im_Bk = ax.scatter(self.MU_pos.cpu()[:, 0], self.MU_pos.cpu()[:, 1], c=SINR.cpu(), marker=".", s=12)  # MUs
        ax.scatter(BS_pos[0].cpu(), BS_pos[1].cpu(), c="red", marker="^", label="BS", s=12)  # BS
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("SINR")
        ax.set_aspect(aspect="equal")
        fig.colorbar(im_Bk, ax=ax)

        fig.tight_layout()
        fig.savefig(f"{dir}/SINR_{self.scenario}.png", dpi=300)
        print(f'Saved: "{dir}/SINR_{self.scenario}.png"')

    """Old Functions"""

    def calcJointCoef(
        self,
        h_Bk: torch.Tensor,
        Z_Theta: torch.Tensor,
        H_BS: torch.Tensor,
        h_Sk: torch.Tensor,
        batch_size: int = None,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.device
        Z_Theta = Z_Theta.to(device)
        K, M = self.K, self.M

        H = torch.zeros((len(Z_Theta), K, M), dtype=torch.cfloat, device=device)
        count = 0
        if progress:  # show progressbar
            pbar = tqdm.tqdm(range(len(Z_Theta)))

        # parallelism
        if batch_size is None:
            batch = len(Z_Theta)
        else:
            batch = min(batch_size, len(Z_Theta))

        while count < len(Z_Theta):
            start = count
            end = min(len(Z_Theta), count + batch)
            # dec ompose to increase gpu memory efficiency of einsum:
            tempMat = torch.einsum("zn,nm->znm", Z_Theta[start:end], H_BS)  # (Z, N, M)
            H[start:end] = h_Bk + torch.einsum("kn,znm->zkm", h_Sk, tempMat)  # (Z, K, M)
            if progress:  # show progressbar
                pbar.update(n=min(batch, len(Z_Theta) - count))
            count += batch
        return H  # (Z, K, M), (K)

    def coef(
        self,
        Z_Theta: torch.Tensor,
        alpha_los: float = 2,
        alpha_nlos: float = 4,
        kapa: float = 10,  # 10 dB
        batch_size: int = None,
        progress: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate the joint channel coefficient matrix and (1/SNR).

        Args:
            Z_Theta: Z*Theta
            alpha_los, alpha_nlos: propagation decay factor (for power)
            kapa: Rician K-factor (no unit)
            batch_size: Used to prevent CUDA out of memory. Decrease this value if CUDA out of memory happens, or a larger value computes faster.
            progress: whether to show progress bar

        Results:
            Channel coefficient H.          (Z, K, M)
            reciprocal of real SNR (1/SNR). (K)
        """
        device = self.device
        self.getBlockMats()
        self.getDistMats()
        eta, zeta = sqrt(kapa / (1 + kapa)), sqrt(1 / (1 + kapa))
        M, K = self.M, self.K

        self.__generate_channels__(eta, zeta, alpha_los, alpha_nlos)

        Z_Theta = Z_Theta.to(device)

        H = torch.zeros((len(Z_Theta), K, M), dtype=torch.cfloat, device=device)
        count = 0
        if progress:  # show progressbar
            pbar = tqdm.tqdm(range(len(Z_Theta)))

        # parallelism
        if batch_size is None:
            batch = len(Z_Theta)
        else:
            batch = min(batch_size, len(Z_Theta))

        while count < len(Z_Theta):
            start = count
            end = min(len(Z_Theta), count + batch)
            # decompose to increase gpu memory efficiency of einsum:
            tempMat = torch.einsum("zn,nm->znm", Z_Theta[start:end], self.H_BS)  # (Z, N, M)
            H[start:end] = self.h_Bk + torch.einsum("kn,znm->zkm", self.h_Sk, tempMat)  # (Z, K, M)
            # H[start:end] = torch.einsum("kn,znm->zkm", self.h_Sk, tempMat)  # (Z, K, M)
            # H[start:end] = self.h_Bk # (Z, K, M)
            if progress:  # show progressbar
                pbar.update(n=min(batch, len(Z_Theta) - count))
            count += batch
        self.H = H
        AWGN_norm = self.AWGN_norm
        return H, AWGN_norm  # (Z, K, M), (K)

    def mean_coef(
        self,
        Z_Theta: torch.Tensor,
        alpha_los: float = 2,  # 2,
        alpha_nlos: float = 4,  # orig: 2.5, now: 4
        kapa: float = 10,  # 10 dB
        batch_size: int = None,
        progress: bool = False,
        sample_time: int = 1,  # 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the statistical results of self.coef

        Args:
            Z_Theta: Z*Theta
            alpha_los, alpha_nlos: propagation decay factor (for power)
            kapa: Rician K-factor (no unit)
            batch_size: Used to prevent CUDA out of memory. Decrease this value if CUDA out of memory happens, or a larger value computes faster.
            progress: whether to show progress bar
            sample_tim: The time of samples to obtain the mean value. Defaults to 1.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        H_mean, AWGN_norm_mean = 0, 0
        for _ in range(sample_time):
            self.__init_channel__()
            H, AWGN_norm = self.coef(Z_Theta, alpha_los, alpha_nlos, kapa, batch_size, progress)
            H_mean += H
            AWGN_norm_mean += AWGN_norm
        H_mean /= sample_time
        AWGN_norm_mean /= sample_time

        return H_mean, AWGN_norm_mean


class Beamformer:
    def __init__(self, device: str):
        """
        Args:
            device: 'cpu' or 'cuda:<num>'
        """
        self.device = device

    def to(self, device: str):
        """Move tensors onto the specific device.

        Args:
            device: 'cpu' or 'cuda:<num>'

        Returns:
            The object on the new device
        """
        self.device = device
        return self

    def MRT(self, H: torch.Tensor) -> torch.Tensor:
        """Return the MRT (Maximum Rate Transmission) beamformer. (Z, K, M)

        Args:
            H: The channel coefficient matrix. (Z, K, M)
        """
        device = self.device

        H = H.to(device)
        W = H.conj() / H.norm(dim=2, keepdim=True)  # MRT beamformer
        W[H.sum(dim=2) == 0] = 0

        return W


if __name__ == "__main__":
    import os
    import argparse
    from itertools import product
    from random import random

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, dest="scenario", default="small city")
    parser.add_argument("-g", "--gpu", type=str, dest="gpu", default="")

    args = parser.parse_args()
    device = f"cuda{':' + args.gpu if args.gpu != '' else ''}" if torch.cuda.is_available() else "cpu"
    scenario = args.scenario
    # Load scenario parameters
    wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode = scenario_configs(scenario)

    print(f"fc: {3e8 / wavelength * 1e-9:.4f} GHz")
    print(f"scenario : {scenario}")
    print(f"device: {device}")

    # Initialize the channel
    channel = Channel(
        wavelength=wavelength,
        d_ris_elem=d_ris_elem,
        ris_center=ris_center,
        ris_norm=ris_norm,  # z=0 for AABB
        ris_size=ris_size,
        area=area,
        BS_pos=BS_pos,
        M=M,
        K=K,
        device=device,
        MU_dist=MU_mode,
        rand_seed=int(random()*1e3)
    )
    beamformer = Beamformer(device=device)

    channel.config_scenario(centers_=centers, obstacles=obstacles, std_radius=std, scenario_name=scenario)
    channel.create()

    # Test
    # n_env = 1  # number of parallel environments
    n_iter = 1
    h_ris = ris_size[0]
    w_ris = ris_size[1]

    n_bit = 8
    n_env = 2 ** (n_bit * 2)
    RIS_beambook = []
    for theta, phi in product(np.linspace(0, 2 * np.pi, 2**n_bit), np.linspace(0, 2 * np.pi, 2**n_bit)):
        RIS_beambook.append((theta, phi))

    # Z = torch.randint(0, 2, (n_env, h_ris*w_ris)).type(torch.float32).to(device) # blocking condition, a binary matrix
    # Z = torch.zeros((n_env, h_ris*w_ris)).type(torch.float32).to(device) # blocking condition, a binary matrix
    Z = torch.ones((n_env, h_ris * w_ris)).type(torch.float32).to(device)  # blocking condition, a binary matrix
    # Theta = torch.polar(torch.ones_like(Z), 2 * pi * torch.zeros_like(Z))  # phase shift
    Theta = torch.polar(torch.ones_like(Z), torch.zeros_like(Z))  # phase shift
    for idx_beam, (theta, h) in enumerate(RIS_beambook):
        x_steer_vec = torch.polar(
            torch.ones(h_ris), torch.Tensor(2 * np.pi * d_ris_elem / wavelength * np.sin(theta) * np.sin(phi) * np.arange(h_ris))
        )
        y_steer_vec = torch.polar(
            torch.ones(w_ris), torch.Tensor(2 * np.pi * d_ris_elem / wavelength * np.sin(theta) * np.cos(phi) * np.arange(w_ris))
        )
        Theta[idx_beam] = torch.kron(y_steer_vec, x_steer_vec).flatten()
    Theta.to(device)

    B_BS, B_Sk, B_Bk = channel.getBlockMats()
    D_BS, D_Sk, D_Bk = channel.getDistMats()

    channel.update_environment()
    for i in range(n_iter):
        channel.update_channel(alpha_nlos=4)
        # H, AWGN_norm = channel.coef(Z_Theta, batch_size=None, progress=True)
        h_Bk, h_BS, h_Sk, AWGN_norm = channel.get_channel_coefficient()        
        H = channel.get_joint_channel_coefficient(Z * Theta)
        W = beamformer.MRT(H)
        Pr_BS_RIS_UE = 10 * torch.log10(torch.abs(torch.einsum("kn,zn,nm,zkm->zk", h_Sk, Z * Theta, h_BS, W)))
        Pr_BS_UE = 10 * torch.log10(torch.abs(torch.einsum("km,zkm->zk", h_Bk, W)))
        print(f"BS-RIS: {10 * torch.log10(torch.abs(torch.sum(h_BS))).item()} dB")
        print(f"RIS-UE: {10 * torch.log10(torch.abs(torch.sum(h_Sk))).item()} dB")
        # print(f"BS-RIS-UE: {Pr_BS_RIS_UE + 30} dBm")
        # print(f"BS-UE: {Pr_BS_UE} dB")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(range(n_env), Pr_BS_RIS_UE.cpu().numpy(),color="b", label="BS-RIS-UE")
        ax[0].plot(range(n_env), Pr_BS_UE.cpu().numpy(), color="r", label="BS-UE")
        # ax.hlines(Pr_BS_UE.cpu().numpy(), 0, n_env - 1, colors="r", label="BS-UE")
        ax[0].legend(loc="best")
        ax[0].set_title("Channel Gain (dB)")
        ax[0].set_xlabel("beam number")
        ax[0].set_ylabel("Channel gain (dB)")
        ax[0].grid()
        
        ax[1].plot(range(n_env), Pr_BS_RIS_UE.cpu().numpy()-Pr_BS_UE.cpu().numpy(), label="BS-RIS-UE/BS-UE")        
        ax[1].set_title("BS-RIS-UE/BS-UE (dB)")
        ax[1].set_xlabel("beam number")
        ax[1].set_ylabel("Gain (dB)")
        ax[1].grid()
        
        fig.suptitle(f"RIS size: {h_ris * wavelength / 2 * 100:.2f} cm x{w_ris * wavelength / 2 * 100:.2f} cm")
        fig.tight_layout()
        fig.savefig("channel gain.png")

        SINR = channel.SINR(H, W)

        # print(f"SINR: {10 * torch.log10(SINR)} dB")
        # print("B_BS, B_Sk, B_Bk:", B_BS.shape, B_Sk.shape, B_Bk.shape)
        # print("D_BS, D_Sk, D_Bk:", D_BS.shape, D_Sk.shape, D_Bk.shape)
        # print("H:", H.shape, type(H))
        # print("W:", W.shape, type(W))

    dir = os.path.abspath(f"./Scenario/{scenario}")
    if not os.path.isdir(dir):
        os.makedirs(dir)
    channel.plot_block_cond(dir)
    channel.show(dir)
    channel.plot_SINR(dir, SINR[0])  # rarely used
