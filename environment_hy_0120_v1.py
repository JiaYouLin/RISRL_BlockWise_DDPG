import torch
import torch.nn.functional as F
import einops
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import os
import sys
import math
from typing import List, Tuple

"""
Settings
"""
os.chdir(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZE = 65536 # adjust this number to prevent CUDA out of memory.

"""
Constant and  (DO NOT change here)
"""
Vertices = [  # coordinates of 8 vertices
    [-0.5, -0.5, 0],
    [+0.5, -0.5, 0],
    [+0.5, +0.5, 0],
    [-0.5, +0.5, 0],
    [-0.5, -0.5, +1],
    [+0.5, -0.5, +1],
    [+0.5, +0.5, +1],
    [-0.5, +0.5, +1],
]

Faces = np.array([  # index of 6 faces
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [1, 2, 6, 5],
    [0, 4, 7, 3],
    [2, 3, 7, 6],
    [0, 1, 5, 4],
])


def Rot_xy(angle: float, device=None):
    """Rotation matrix that rotate around the z-axis.

    Args:
        angle: The angle to rotate. (unit: degree)

    Returns:
        The corresponding 3x3 z-rotation matrix.
    """
    angle = math.radians(angle)
    cos, sin = math.cos(angle), math.sin(angle)
    R_xy = torch.tensor([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])

    if device is not None:
        R_xy = R_xy.to(device)
    return R_xy


class Obstacle:
    def __init__(
        self,
        pos=(0, 0),
        size=(10, 10),
        height=10,
        rotate=0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Create an object of cuboid obstacles.

        Args:
            pos: Center of the obstacle on the xy-plane. (x_c, y_c)
            size: The size of the obstacle on xy-plane. (d_x, d_y)
            height: The height of the obstacle. d_z
            rotate: The counter-clockwise rotation angle of the obstacle around its center. (unit: degree)
            device: 'cpu' or 'cuda:<num>'

        Returns:
            An Obstacle object.
        """
        # Checkings
        assert len(pos) == 2
        assert len(size) == 2
        assert height > 0

        self.pos = torch.tensor(pos + (0,))
        self.size = torch.tensor(size + (height,))
        self.rotate = rotate

        self.verts = None
        self.norm = None
        self.c = None  # constant of face equations
        self.poly_list = None
        self.device = device

    def to(self, device: str):
        """Move tensors onto the specific device.

        Args:
            device: 'cpu' or 'cuda:<num>'

        Returns:
            Original object on the new device
        """
        self.device = device
        self.pos = self.pos.to(device).type(torch.float)
        self.size = self.size.to(device).type(torch.float)
        return self

    def vertices(self) -> torch.Tensor:
        """Generate the coordinate of the 8 vertices.

        Returns:
            A 8x3 Tensor. (vertex, (x,y,z))
        """
        device = self.device

        if self.verts is None:
            R_xy = Rot_xy(self.rotate).to(device).type(torch.float)
            self.verts = torch.tensor(Vertices, dtype=torch.float, device=device)
            self.verts = torch.einsum("qp,vp,p->vq", R_xy, self.verts, self.size)
            self.verts = einops.repeat(self.pos, "p -> v p", v=8) + self.verts
        return self.verts

    def norm_vec(self) -> torch.Tensor:
        """Generate the normal vectors of 6 faces.

        Returns:
            A 6x3 Tensor. (face, (x,y,z))
        """
        if self.norm is None:
            device = self.device

            self.vertices()
            self.norm = torch.zeros(size=(6, 3), device=device)
            self.c = torch.zeros(size=(6, 1), device=device)
            for i, face in enumerate(Faces):
                v1 = self.verts[face[1], :] - self.verts[face[0], :]
                v2 = self.verts[face[2], :] - self.verts[face[1], :]
                self.norm[i, :] = F.normalize(torch.cross(v1, v2, 0), dim=0)
                self.c[i, :] = torch.sum(self.norm[i, :] * self.verts[face[0], :], dim=0)
        return self.norm

    def polies(self) -> List[Poly3DCollection]:
        """Create a list of 3D polygons. These polies are used for scenario visualization.

        Returns:
            The list of 3D polygons.
        """
        if self.poly_list is None:
            self.poly_list = []
            self.vertices()
            for face in Faces:
                poly = Poly3DCollection(verts=[list(self.verts[face, :].cpu().numpy())], facecolors="orange", edgecolors="darkorange", linewidths=1, alpha=0.5)
                self.poly_list.append(poly)
        return self.poly_list

    def proj2D(self) -> torch.Tensor:
        """Project the vertices onto the xy-plane.

        Returns:
            A 4x2 Tensor. (vertex, (x,y))
        """
        self.vertices()
        return self.verts[Faces[0], :2]

    def isInside(self, pt: torch.Tensor) -> torch.BoolTensor:
        """Check whether each of N points on the xy-plane is inside the projection of the obstacle.

        Args:
            pt: The points. (N, (x,y))

        Returns:
            A boolean Tensor indicating Whether each point is inside the projection of the obstacle. (N)
        """
        device = self.device

        self.vertices()
        pt = pt.to(device)
        pt = pt if len(pt.shape) > 1 else pt.unsqueeze(0)

        v = []  # sides
        u = []  # v_p
        for i in range(4):
            v.append(self.verts[Faces[0, (i + 1) % 4], :2] - self.verts[Faces[0, i], :2].unsqueeze(0))  # (s,p)
            u.append(pt.unsqueeze(1) - self.verts[Faces[0, i], :2])  # (b,s,p)
        v = torch.cat(v, dim=0).type(torch.float)  # (s, p)
        u = torch.cat(u, dim=1).type(torch.float)  # (b,s,p)
        isin = torch.einsum("sp,bsp->bs", v, u)  # dot product
        isin = torch.all(torch.where(isin >= 0, 1, 0.0), dim=1, keepdim=False)  # inside the projection if all values are positive
        return isin

    def isInside3D(self, pt: torch.Tensor) -> torch.BoolTensor:
        """Check whether each of N points is inside the obstacle.
        This function is implemented by checking the sign of inner products from the point to pivots of the projection.
        Args:
            pt: The points. (N, (x,y,z))

        Returns:
            A boolean Tensor indicating whether each point is inside the projection of the obstacle. (N)
        """
        device = self.device

        self.vertices()
        pt = pt.to(device)
        pt = pt if len(pt.shape) > 1 else pt.unsqueeze(0)
        n_CH = len(pt)

        isinZs = torch.logical_and(
            torch.ge(pt[:, 2], self.verts[Faces[0, 0], 2] * torch.ones(size=(n_CH,), device=device)),
            torch.ge(self.verts[Faces[1, 0], 2] * torch.ones(size=(n_CH,), device=device), pt[:, 2]),
        )
        isin3D = torch.logical_and(self.isInside(pt[:, :2]), isinZs)

        return isin3D

    def isIntersect(self, pt1: torch.Tensor, pt2: torch.Tensor) -> torch.BoolTensor:
        """Check if N line segments (p1, p2) intersect with the obstacle.
        Leverage Slabs Method to a AABB cube.

        Args:
        pt1, pt2: The points compose the line segment. (N, (x,y,z)).
        """
        device = self.device
        self.norm_vec()

        # s = pt1.type(torch.float32)
        # d = pt2.type(torch.float32)
        s = pt1 if len(pt1.shape) > 1 else pt1.unsqueeze(0)
        d = pt2 if len(pt2.shape) > 1 else pt2.unsqueeze(0)

        n_CH = len(s)
        c = einops.repeat(self.c, "f p -> b f p", b=n_CH)  # p=1
        n = einops.repeat(self.norm, "f p -> b f p", b=n_CH)
        O = einops.repeat(s, "b p -> b f p", f=6)
        D = einops.repeat(F.normalize(d - s, dim=1), "b p -> b f p", f=6)

        # Check intersections with planes that dot(n, D) != 0
        t = (c.squeeze(2) - torch.sum(O * n, dim=2)) / torch.sum(D * n, dim=2)
        t_inf = torch.where(torch.logical_or(t.isneginf(), t.isnan()), np.inf * torch.ones_like(t, device=device), t)
        t_neginf = torch.where(torch.logical_or(t.isinf(), t.isnan()), -np.inf * torch.ones_like(t, device=device), t)

        tx_min, _ = torch.min(t_neginf[:, 2:4], 1, keepdim=True)
        tx_max, _ = torch.max(t_inf[:, 2:4], 1, keepdim=True)
        ty_min, _ = torch.min(t_neginf[:, 4:6], 1, keepdim=True)
        ty_max, _ = torch.max(t_inf[:, 4:6], 1, keepdim=True)
        tz_min, _ = torch.min(t_neginf[:, 0:2], 1, keepdim=True)
        tz_max, _ = torch.max(t_inf[:, 0:2], 1, keepdim=True)

        t_min, _ = torch.max(torch.cat([tx_min, ty_min, tz_min], 1), dim=1)
        t_max, _ = torch.min(torch.cat([tx_max, ty_max, tz_max], 1), dim=1)
        isintersect = torch.ge(t_max, t_min)

        # Check intersections with planes that dot(n, D) == 0
        proj_near = O + einops.repeat(t_min, "b -> b f p", f=6, p=3) * D
        proj_far = O + einops.repeat(t_max, "b -> b f p", f=6, p=3) * D
        c_proj_near = torch.norm(n * proj_near, dim=2)  # (n_CH, 6)
        c_proj_far = torch.norm(n * proj_far, dim=2)  # (n_CH, 6)

        cx_min, _ = torch.min(c.view(n_CH, 6)[:, 2:4], 1)  # (n_CH)
        cx_max, _ = torch.max(c.view(n_CH, 6)[:, 2:4], 1)  # (n_CH)
        cy_min, _ = torch.min(c.view(n_CH, 6)[:, 4:6], 1)  # (n_CH)
        cy_max, _ = torch.max(c.view(n_CH, 6)[:, 4:6], 1)  # (n_CH)
        cz_min, _ = torch.min(c.view(n_CH, 6)[:, 0:2], 1)  # (n_CH)
        cz_max, _ = torch.max(c.view(n_CH, 6)[:, 0:2], 1)  # (n_CH)

        inter_x = torch.logical_and(
            torch.le((c_proj_near[:, 2] - cx_min) * (c_proj_near[:, 2] - cx_max), torch.zeros((n_CH), device=device)),
            torch.le((c_proj_far[:, 2] - cx_min) * (c_proj_far[:, 2] - cx_max), torch.zeros((n_CH), device=device)),
        )
        inter_y = torch.logical_and(
            torch.le((c_proj_near[:, 4] - cy_min) * (c_proj_near[:, 4] - cy_max), torch.zeros((n_CH), device=device)),
            torch.le((c_proj_far[:, 4] - cy_min) * (c_proj_far[:, 4] - cy_max), torch.zeros((n_CH), device=device)),
        )
        inter_z = torch.logical_and(
            torch.le((c_proj_near[:, 0] - cz_min) * (c_proj_near[:, 0] - cz_max), torch.zeros((n_CH), device=device)),
            torch.le((c_proj_far[:, 0] - cz_min) * (c_proj_far[:, 0] - cz_max), torch.zeros((n_CH), device=device)),
        )

        # ignore this judgement if dot(n, D) !=0 to avoid unnecessary errors
        inter_x = torch.where(tx_max.squeeze(1).isinf(), inter_x, isintersect)
        inter_y = torch.where(ty_max.squeeze(1).isinf(), inter_y, isintersect)
        inter_z = torch.where(tz_max.squeeze(1).isinf(), inter_z, isintersect)

        areintercube = torch.logical_and(inter_z, torch.logical_and(inter_y, inter_x)).bool()  # Whether the instersections are inside or on the cube.

        # Handle segment intersection
        t_d = torch.norm(d - s, dim=1)
        areout_left = torch.logical_and(torch.gt(t_min, t_d), torch.ge(t_d, torch.zeros_like(t_d, device=device)))
        areout_right = torch.logical_and(torch.gt(torch.zeros_like(t_max, device=device), t_max), torch.ge(t_d, torch.zeros_like(t_max, device=device)))
        areout = torch.logical_or(areout_left, areout_right)

        # Hybrid the results
        isintersect = torch.logical_and(torch.logical_and(isintersect, areintercube), ~areout)
        return isintersect


class Environment:
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
        rand_seed: int,
    ):
        """The object describing the simulation environment.

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
            rand_seed: seed for random number generator

        Returns:
            An Environment object.
        """
        self.device = device
        self.M = M
        self.MU_dist = MU_dist
        self.K_expect = K
        self.N = ris_size[0] * ris_size[1]
        self.area = area
        self.obstacles: Obstacle = None
        self.centers: List[Tuple[float]] = None
        self.wavelength = wavelength
        self.d = d_ris_elem

        self.BS_pos = torch.tensor(BS_pos)

        self.ris_center = ris_center
        self.ris_size = ris_size
        self.ris_norm: torch.Tensor = F.normalize(torch.tensor(ris_norm, dtype=torch.float32), dim=0)
        self.ris_pos: torch.Tensor = None

        # Check
        assert len(ris_norm) == 3 # (x, y, z)
        assert len(ris_size) == 2 # (h, w) in RIS elements
        assert len(area) == 2 # (x, y)
        assert len(BS_pos) == 3 # (x, y, z)

        assert ris_norm[2] == 0
        assert wavelength > 0
        assert ris_center[2] >= self.d * ris_size[0] / 2
        assert area[0] > 0 and area[1] > 0
        
        assert BS_pos[0] > (-area[0] / 2) and BS_pos[0] < (area[0] / 2)
        assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        assert BS_pos[2] >= 0
        assert M > 0
        
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(rand_seed)

        self.__init_environment__()
        self.to(device)

    def to(self, device: str):
        """Move tensors onto the specific device.

        Args:
            device: 'cpu' or 'cuda:<num>'

        Returns:
            The object on the new device
        """
        self.device = device
        self.BS_pos = self.BS_pos.to(device)
        self.ris_norm = self.ris_norm.to(device)

        if self.ris_pos is not None:
            self.ris_pos = self.ris_pos.to(device)
        if self.MU_pos is not None:
            self.MU_pos = self.MU_pos.to(device)
        if self.BS_antenna_pos is not None:
            self.BS_antenna_pos = self.BS_antenna_pos.to(device)
        return self

    def __init_environment__(self):
        self.B_BS, self.B_Sk, self.B_Bk = None, None, None
        self.D_BS, self.D_Sk, self.D_Bk = None, None, None
        self.BS_antenna_pos = None
        self.ris_pos = None
        self.MU_pos = None
        
        return self
    
    def update_environment(self):
        self.__init_environment__()
        self.create()
    
    def config_scenario(
        self,
        std_radius: float,  # std for 2D Gaussian distribution/ radius for Poisson point process
        scenario_name: str,
        obstacles: List[Obstacle],
        centers_: List[Tuple[float]],  # normal
    ) -> None:
        """Configure the simulation scenario variables.

        Args:
            std_radius: If the MU distribution self.MU_dist follows ppp (poisson), this is radius of each regioon; if self.MU_dist is 2D Gaussian (normal), this is std. 
            scenario_name: Name of scenario used in dumped filename.
            obstacles: The list of Obstacles.
            centers_: Center of MU groups. [(x1, y1), (x2, y2), ...]
        """
        device = self.device
        self.scenario = scenario_name
        self.obstacles: List[Obstacle] = []

        # devices are clustered and modeled as Poisson point process or 2D Gaussian
        assert (centers_ is not None) and (len(centers_) > 0)
        assert std_radius > 0
        self.centers = centers_
        self.std_radius = std_radius

        assert len(obstacles) > 0
        self.obstacles = obstacles

        for obs in self.obstacles:  # put the obstacles onto device
            obs.to(device)

    def __init_BS__(self) -> None:
        """Initialize the position of BS antennas."""
        if self.BS_antenna_pos is None:
            device = self.device
            M = self.M
            self.BS_antenna_pos = einops.repeat(self.BS_pos,'c -> m c', m=self.M).to(device) # (M, (x,y,z))
            d_ant = self.wavelength/2
            self.d_ant = d_ant
            
            # BS antenna is a UPA
            Mx, My = 1, M
            for i in range(1, int(M**0.5) + 1):
                if M % i == 0:
                    Mx, My = M//i, i
            ant_pos_x = torch.zeros(Mx, My)
            ant_pos_y, ant_pos_z = torch.meshgrid(torch.arange(Mx) - 0.5 * (Mx-1), torch.arange(My) - 0.5 * (My-1), indexing='ij')
            self.BS_antenna_pos = self.BS_antenna_pos  + d_ant * torch.cat([ant_pos_x.reshape(-1, 1), ant_pos_y.reshape(-1, 1), ant_pos_z.reshape(-1, 1)], dim=1).to(device)

    def __init_RIS__(self) -> None:
        """Initialize the position of RIS elements."""
        if self.ris_pos is None:
            device = self.device

            # RIS elements
            d = self.d
            ris_size = self.ris_size
            ris_norm = self.ris_norm
            ris_center = torch.tensor(self.ris_center, device=device) # (-area[0] / 2, 0, ris_height + d * ris_size[0] / 2)

            x, y = torch.meshgrid(
                torch.arange(-ris_size[0] * d / 2, ris_size[0] * d / 2, d),
                torch.arange(-ris_size[1] * d / 2, ris_size[1] * d / 2, d),
                indexing="ij",
            )
            x = x.to(device).type(torch.float32)
            y = y.to(device).type(torch.float32)
            z = torch.zeros(size=ris_size, device=device)
            self.ris_pos = torch.cat([x.unsqueeze(2), y.unsqueeze(2), z.unsqueeze(2)], dim=2).to(device)
            n_xy_unit = F.normalize(ris_norm[:2].clone().detach().to(device), dim=0)

            deg_xy = math.degrees(math.atan(n_xy_unit[1] / n_xy_unit[0]))
            R_xy = Rot_xy(deg_xy).to(device)  # rotation matrix on x-y
            n_unit = F.normalize(ris_norm.clone().detach().to(device), dim=0)
            z_unit = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
            u_unit = F.normalize(torch.linalg.cross(z_unit, n_unit), dim=0)
            cos = torch.dot(z_unit, n_unit)
            sin = math.sqrt(1 - cos**2)
            u_cross_prod_mat = torch.tensor([[0, -u_unit[2], u_unit[1]], [u_unit[2], 0, -u_unit[0]], [-u_unit[1], u_unit[0], 0]], device=device) # cross product matrix
            R = cos * torch.eye(3, device=device) + sin * u_cross_prod_mat + (1 - cos) * torch.outer(u_unit, u_unit)  # rotation matrix
            self.ris_pos = torch.matmul(R @ R_xy.unsqueeze(0), self.ris_pos.view(-1, 3, 1)).reshape(ris_size[0], ris_size[1], 3)
            self.ris_pos = self.ris_pos - torch.mean(self.ris_pos, dim=[0, 1]) + ris_center
    
    def __init_MUs__(self) -> None:
        """Initialize user positions."""
        if self.MU_pos is None:
            device = self.device
            while (self.MU_pos is None) or (len(self.MU_pos) < self.K_expect):          # 當 self.MU_pos 的長度小於期望的UE數量 self.K_expect 時, 會進入迴圈重新生成UE位置, 直到達到預期的數量
                if self.MU_dist == "poisson":  # Poisson point process
                    
                    N = torch.poisson(
                        (self.K_expect / len(self.centers))* torch.ones(
                            len(self.centers),
                            device=device,
                        ), generator=self.generator,
                    ).type(torch.int)

                    UE_num = int(torch.sum(N).item())
                    radius = torch.rand(UE_num, generator=self.generator, device=device) * self.std_radius
                    angle = torch.rand(UE_num, generator=self.generator, device=device) * (2 * math.pi)
                    x = torch.zeros((UE_num,), device=device)
                    y = torch.zeros((UE_num,), device=device)

                    itr_ = 0
                    for g_idx in range(len(self.centers)):
                        x[itr_ : itr_ + N[g_idx]] = self.centers[g_idx][0] + radius[itr_ : itr_ + N[g_idx]] * torch.cos(angle[itr_ : itr_ + N[g_idx]])
                        y[itr_ : itr_ + N[g_idx]] = self.centers[g_idx][1] + radius[itr_ : itr_ + N[g_idx]] * torch.sin(angle[itr_ : itr_ + N[g_idx]])
                        itr_ += N[g_idx].item()
                elif self.MU_dist == "normal":  # Normal dstribution

                    N = self.K_expect // len(self.centers)
                    x = torch.zeros(N * len(self.centers), device=device)
                    y = torch.zeros(N * len(self.centers), device=device)

                    itr_ = 0
                    for g_idx in range(len(self.centers)):
                        x[itr_ : itr_ + N] = torch.normal(mean=self.centers[g_idx][0], std=self.std_radius, size=(N,), generator=self.generator, device=device)
                        y[itr_ : itr_ + N] = torch.normal(mean=self.centers[g_idx][1], std=self.std_radius, size=(N,), generator=self.generator, device=device)
                        itr_ += N
                else:
                    print(f"Invalid distribution name: {self.MU_dist}.")
                    exit()

                x = x.clip(-self.area[0] / 2, self.area[0] / 2)
                y = y.clip(-self.area[1] / 2, self.area[1] / 2)

                MU_pos = torch.cat([x.unsqueeze(1), y.unsqueeze(1)], dim=1).to(device)

                isInObstacles = torch.zeros(size=(len(MU_pos),), device=device)
                for obs in self.obstacles:
                    isInObstacles = torch.logical_or(isInObstacles, obs.isInside(MU_pos))

                included = ~isInObstacles

                if self.MU_pos is None:
                    self.MU_pos = MU_pos[included, :]
                else:
                    self.MU_pos = torch.cat([self.MU_pos, MU_pos[included, :]], dim=0)
            self.MU_pos = self.MU_pos[:self.K_expect]
            self.K = len(self.MU_pos)
            assert self.K == self.K_expect
            # print(f"Number of MUs: {self.K}")

    def __is_left__(self, p1s_: torch.Tensor, p2s_: torch.Tensor, pts_: torch.Tensor) -> torch.BoolTensor:
        """Helping function of __is_in_range__.
        * Winding number algorithm: https://saturncloud.io/blog/integer-winding-number-algorithm-with-edge-cases/

        Args:
            p1s_: (num_pt, K, 2).
            p2s_: (num_pt, K, 2).
            pts_: (num_pt, K, 2).

        Returns:
            (num_pt, K, 1)
        """
        device = self.device
        p1s = p1s_.to(device)
        p2s = p2s_.to(device)
        pts = pts_.to(device)

        # (p2[0] - p1[0]) * (point[1] - p1[1]) - (point[0] - p1[0]) * (p2[1] - p1[1])
        isLeft = (p2s[:, :, 0] - p1s[:, :, 0]) * (pts[:, :, 1] - p1s[:, :, 1]) - (pts[:, :, 0] - p1s[:, :, 0]) * (p2s[:, :, 1] - p1s[:, :, 1])
        return isLeft

    def __MU_RIS_large_MatMul_strategy_selection__(self) -> Tuple[float]:
        """Select the best strategy based on time*space used for computation. This function exists because number of users K and number of RIS elements are usually large.
        """
        if BATCH_SIZE >= self.K:
            batch_K = math.ceil(BATCH_SIZE / self.K)
            t_K = math.ceil(self.N / batch_K)
            s_K = batch_K * self.K
        else:
            batch_K = math.ceil(self.K / BATCH_SIZE)
            t_K = self.N * batch_K
            s_K = BATCH_SIZE
        ts_K = t_K * s_K

        if BATCH_SIZE >= self.N:
            batch_N = math.ceil(BATCH_SIZE / self.N)
            t_N = math.ceil(self.K / batch_N)
            s_N = batch_N * self.N
        else:
            batch_N = math.ceil(self.N / BATCH_SIZE)
            t_N = self.K * batch_N
            s_N = BATCH_SIZE
        ts_N = t_N * s_N

        return ts_K, ts_N, batch_K, batch_N

    def create(self) -> None:
        """Create the whole simulation environment."""
        if self.obstacles is None:
            print("Please run config_scenario() beforehand.", file=sys.stderr)
        self.__init_BS__()
        self.__init_RIS__()
        self.__init_MUs__()
    
    def getBlockMats(self) -> Tuple[torch.Tensor]:
        """Return three visibility (binary) matrices: B_BS, B_Sk, and B_Bk. If no obstacle on the path, set to 1, otherwise 0.

        Returns:
            B_BS: Visibility condition between BS and RIS. (N, M)
            B_Sk: Visibility condition between RIS element n and user k. (K, N)
            B_Bk: Visibility condition between BS and user k. (K, M)
        """
        if (self.B_BS is None) or (self.B_Sk is None) or (self.B_Bk is None):
            device = self.device
            self.create()

            B_BS = torch.zeros(size=(self.N, self.M), device=device).bool().flatten(end_dim=1)
            B_Sk = torch.zeros(size=(self.K, self.N), device=device).bool().flatten(end_dim=1)
            B_Bk = torch.zeros(size=(self.K, self.M), device=device).bool().flatten(end_dim=1)

            MU_pos = torch.cat([self.MU_pos, torch.zeros(size=(self.K, 1), device=device)], dim=1)

            # BS-RIS
            BS_bs_pos = einops.repeat(self.BS_antenna_pos, 'm p -> n m p', n=self.N)
            RIS_bs_pos = einops.repeat(self.ris_pos.view(-1,3), 'n p -> n m p', m=self.M)
            for obs in self.obstacles:
                B_BS = torch.logical_or(B_BS, obs.isIntersect(BS_bs_pos.flatten(0, 1), RIS_bs_pos.flatten(0, 1)))
            B_BS = (~B_BS).unflatten(dim=0, sizes=(self.N, self.M))

            # RIS-MU
            ts_K, ts_N, batch_K, batch_N = self.__MU_RIS_large_MatMul_strategy_selection__()

            if ts_N >= ts_K:
                if BATCH_SIZE >= self.K:
                    length = self.N * self.K
                    batch = batch_K * self.K
                    count = 0
                    while (length - count * batch) > 0:
                        LB = count * batch
                        UB = min(length, (count + 1) * batch)
                        LB_N = LB // self.K
                        UB_N = UB // self.K

                        RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3)[LB_N:UB_N], 'n p -> n k p', k=self.K).flatten(end_dim=1)
                        MU_Sk_pos = einops.repeat(MU_pos, 'k p -> n k p', n=UB_N-LB_N).flatten(end_dim=1)

                        for obs in self.obstacles:
                            B_Sk[LB:UB] = torch.logical_or(B_Sk[LB:UB], obs.isIntersect(RIS_Sk_pos, MU_Sk_pos))
                        count += 1
                else:
                    for i in range(self.N):
                        length = self.K
                        batch = BATCH_SIZE
                        count = 0
                        while (length - count * batch) > 0:
                            LB = count * batch
                            UB = min(length, (count + 1) * batch)

                            RIS_Sk_pos = einops.repeat(self.ris_pos[i].view(-1, 3)[i].unsqueeze(0), 'n p -> n k p', k=UB-LB).flatten(end_dim=1)
                            MU_Sk_pos = einops.repeat(MU_pos[LB:UB], 'k p -> n k p', n=1).flatten(end_dim=1)
                            
                            for obs in self.obstacles:
                                B_Sk[i*self.K + LB:i*self.K + UB] = torch.logical_or(B_Sk[i*self.K + LB:i*self.K + UB], obs.isIntersect(RIS_Sk_pos, MU_Sk_pos))
                            count += 1
                B_Sk = (~B_Sk).unflatten(dim=0, sizes=(self.N, self.K))
                B_Sk = einops.rearrange(B_Sk, 'n k-> k n')
            else:
                if BATCH_SIZE >= self.N:
                    length = self.N * self.K
                    batch = batch_N * self.N
                    count = 0
                    while (length - count * batch) > 0:
                        LB = count * batch
                        UB = min(length, (count + 1) * batch)
                        LB_K = LB // self.N
                        UB_K = UB // self.N

                        RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3), 'n p -> n k p', k=UB_K-LB_K).flatten(end_dim=1)
                        MU_Sk_pos = einops.repeat(MU_pos[LB_K:UB_K], 'k p -> n k p', n=self.N).flatten(end_dim=1)

                        for obs in self.obstacles:
                            B_Sk[LB:UB] = torch.logical_or(B_Sk[LB:UB], obs.isIntersect(RIS_Sk_pos, MU_Sk_pos))
                        count += 1
                else:
                    for j in range(self.K):
                        length = self.N
                        batch = BATCH_SIZE
                        count = 0
                        while (length - count * batch) > 0:
                            LB = count * batch
                            UB = min(length, (count + 1) * batch)

                            RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3)[LB:UB], 'n p -> n k p', k=1).flatten(end_dim=1)
                            MU_Sk_pos = einops.repeat(MU_pos[j].unsqueeze(0), 'k p -> n k p', n=UB-LB).flatten(end_dim=1)
                            
                            for obs in self.obstacles:
                                B_Sk[j*self.N + LB:j*self.N + UB] = torch.logical_or(B_Sk[j*self.N + LB:j*self.N + UB], obs.isIntersect(RIS_Sk_pos, MU_Sk_pos))
                            count += 1
                B_Sk = (~B_Sk).unflatten(dim=0, sizes=(self.K, self.N))

            # BS-MU
            BS_bk_pos = einops.repeat(self.BS_antenna_pos, 'm p -> k m p', k=self.K)
            MU_bk_pos = einops.repeat(MU_pos, 'k p -> k m p', m=self.M)
            for obs in self.obstacles:
                B_Bk = torch.logical_or(B_Bk, obs.isIntersect(BS_bk_pos.flatten(0, 1), MU_bk_pos.flatten(0, 1)))
            B_Bk = (~B_Bk).unflatten(dim=0, sizes=(self.K, self.M))

            self.B_BS, self.B_Sk, self.B_Bk = B_BS, B_Sk, B_Bk
        return self.B_BS, self.B_Sk, self.B_Bk

    def getDistMats(self) -> Tuple[torch.Tensor]:
        """Return three distance matrices: D_BS, D_Sk, D_Bk.

        Results:
            D_BS: Distance between BS and RIS. (N, M)
            D_Sk: Distance between RIS and MU k. (K, N)
            D_Bk: Distance between BS and MU k. (K, M)

        """
        if (self.D_BS is None):
            device = self.device
            ris_center = torch.tensor(self.ris_center, device=self.device)
            self.create()
            # BS-RIS
            BS_bs_pos = einops.repeat(self.BS_antenna_pos, 'm p -> n m p', n=self.N)
            RIS_bs_pos = einops.repeat(self.ris_pos.view(-1,3), 'n p -> n m p', m=self.M)
            D_BS = torch.norm(BS_bs_pos - RIS_bs_pos, dim=2)             
            self.D_BS = D_BS    # (N, M)
            
            r = self.BS_antenna_pos - ris_center
            Dc_BSc = torch.norm(r, dim=1)   # (1, M) 
            Dc_BSc = torch.reshape(Dc_BSc, [1, -1])
            # self.Dc_BSc = torch.reshape(Dc_BSc, [1, -1])
            self.Dc_BcSc = torch.norm(self.BS_pos - ris_center, dim=0)  # (1)
            self.Phi_BSc = r[:, 0].view(1, -1)/Dc_BSc # (1, M)
            
            # r = self.ris_pos.view(-1, 3) - self.BS_pos
            # Dc_BcS =  torch.norm(r, dim=1)   # (N, 1) 
            # self.Dc_BcS = torch.reshape(Dc_BcS, [-1, 1])
            # self.Phi_BcS = torch.reshape(-r[:, 0]/Dc_BcS, [-1, 1])
            
            # incident angle
            r = self.ris_pos.view(-1, 3) - self.BS_pos
            self.cos_angle_in = torch.sum(r * torch.Tensor([[1, 0, 0]]).to(device), dim=1, keepdim=True) / torch.norm(r, dim=1, keepdim=True)
            
        if (self.D_Sk is None):
            ris_center = torch.tensor(self.ris_center, device=self.device)
            self.create()
            MU_pos = torch.cat([self.MU_pos, torch.zeros(size=(self.K, 1), device=self.device)], dim=1)

            # RIS-MU
            D_Sk = torch.zeros(size=(self.K, self.N), device=self.device).flatten(end_dim=1)
            ts_K, ts_N, batch_K, batch_N = self.__MU_RIS_large_MatMul_strategy_selection__()

            if ts_N >= ts_K:
                if BATCH_SIZE >= self.K:
                    length = self.N * self.K
                    batch = batch_K * self.K
                    count = 0
                    while (length - count * batch) > 0:
                        LB = count * batch
                        UB = min(length, (count + 1) * batch)
                        LB_N = LB // self.K
                        UB_N = UB // self.K

                        RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3)[LB_N:UB_N], 'n p -> n k p', k=self.K).flatten(end_dim=1)
                        MU_Sk_pos = einops.repeat(MU_pos, 'k p -> n k p', n=UB_N-LB_N).flatten(end_dim=1)

                        D_Sk[LB:UB] = torch.norm(RIS_Sk_pos - MU_Sk_pos, dim=1)
                        count += 1
                else:
                    for i in range(self.N):
                        length = self.K
                        batch = BATCH_SIZE
                        count = 0
                        while (length - count * batch) > 0:
                            LB = count * batch
                            UB = min(length, (count + 1) * batch)

                            RIS_Sk_pos = einops.repeat(self.ris_pos[i].view(-1, 3)[i].unsqueeze(0), 'n p -> n k p', k=UB-LB).flatten(end_dim=1)
                            MU_Sk_pos = einops.repeat(MU_pos[LB:UB], 'k p -> n k p', n=1).flatten(end_dim=1)
                            
                            D_Sk[i*self.K + LB:i*self.K + UB] = torch.norm(RIS_Sk_pos - MU_Sk_pos, dim=1)
                            count += 1
                D_Sk = D_Sk.unflatten(dim=0, sizes=(self.N, self.K))
                D_Sk = einops.rearrange(D_Sk, 'n k-> k n')
            else:
                if BATCH_SIZE >= self.N:
                    length = self.N * self.K
                    batch = batch_N * self.N
                    count = 0
                    while (length - count * batch) > 0:
                        LB = count * batch
                        UB = min(length, (count + 1) * batch)
                        LB_K = LB // self.N
                        UB_K = UB // self.N

                        RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3), 'n p -> n k p', k=UB_K-LB_K).flatten(end_dim=1)
                        MU_Sk_pos = einops.repeat(MU_pos[LB_K:UB_K], 'k p -> n k p', n=self.N).flatten(end_dim=1)

                        D_Sk[LB:UB] = torch.norm(RIS_Sk_pos - MU_Sk_pos)
                        count += 1
                else:
                    for j in range(self.K):
                        length = self.N
                        batch = BATCH_SIZE
                        count = 0
                        while (length - count * batch) > 0:
                            LB = count * batch
                            UB = min(length, (count + 1) * batch)

                            RIS_Sk_pos = einops.repeat(self.ris_pos.view(-1, 3)[LB:UB], 'n p -> n k p', k=1).flatten(end_dim=1)
                            MU_Sk_pos = einops.repeat(MU_pos[j].unsqueeze(0), 'k p -> n k p', n=UB-LB).flatten(end_dim=1)
                            
                            D_Sk[j*self.N + LB:j*self.N + UB] = torch.norm(RIS_Sk_pos - MU_Sk_pos)
                            count += 1
                D_Sk = D_Sk.unflatten(dim=0, sizes=(self.K, self.N))

            self.D_Sk = D_Sk    # (K, N)
            r = MU_pos - ris_center
            Dc_Sk = torch.norm(r, dim=1, keepdim=True) # (K, 1)
            self.Dc_Sk = Dc_Sk
            self.Phi_Sk = r[:, 0].view(-1, 1)/Dc_Sk # (K, 1)
            
        
        if (self.D_Bk is None):         
            self.create()
            MU_pos = torch.cat([self.MU_pos, torch.zeros(size=(self.K, 1), device=self.device)], dim=1)   
            # BS-MU
            BS_bk_pos = einops.repeat(self.BS_antenna_pos, 'm p -> k m p', k=self.K)
            MU_bk_pos = einops.repeat(MU_pos, 'k p -> k m p', m=self.M)
            D_Bk = torch.norm(BS_bk_pos - MU_bk_pos, dim=2)

            self.D_Bk = D_Bk    # (K, M)
            
            r = MU_pos - self.BS_pos
            Dc_Bk = torch.norm(r, dim=1, keepdim=True) # (K, 1)
            self.Dc_Bk = Dc_Bk
            self.Phi_Bk = torch.reshape(-r[:, 0].view(-1, 1)/Dc_Bk, [-1, 1])    # (K, 1)
            
        return self.D_BS, self.D_Sk, self.D_Bk

    def plot_block_cond(self, dir: str) -> None:
        """Plot the blocking condition of BS-MU and RIS-MU, respectively, from the top view. Besides, plot blocking codition of BS-RIS and RIS-MU on RIS.

        Args:
            dir: The directory to dump the figure.
            batch: Used to prevent CUDA out of memory. Decrease this value if CUDA out of memory happens, or a larger value computes faster.
        """
        self.getBlockMats()

        BS_pos = self.BS_pos
        d = self.d

        fig, ax = plt.subplots(2, 2)

        # RIS
        risproj1, _ = torch.min(self.ris_pos.view(-1, 3)[:, :2], 0)
        risproj2, _ = torch.max(self.ris_pos.view(-1, 3)[:, :2], 0)
        p = torch.cat([risproj1.unsqueeze(0), risproj2.unsqueeze(0)], dim=0)
        if (risproj1 - risproj2)[0] * (risproj1 - risproj2)[0] < 0:
            p = torch.cat([risproj1.unsqueeze(0), risproj2.flip([0]).unsqueeze(0)], dim=0)
        p = p.cpu().numpy()
        ax[0, 0].plot(p[:, 0], p[:, 1], c="royalblue", label="RIS", linewidth=1)
        ax[0, 1].plot(p[:, 0], p[:, 1], c="royalblue", label="RIS", linewidth=1)

        for obs in self.obstacles:  # obstacles
            base = torch.cat([obs.proj2D(), obs.proj2D()[0].unsqueeze(0)], dim=0).cpu().numpy()
            ax[0, 0].plot(base[:, 0], base[:, 1], c="darkorange", label="Obstacles", linewidth=1)
            ax[0, 1].plot(base[:, 0], base[:, 1], c="darkorange", label="Obstacles", linewidth=1)

        # BS-MU (K, M)
        Bk_deg = torch.mean(self.B_Bk.type(torch.float32), dim=1)  # degree of the blocking condition on BS-MU
        im_Bk = ax[0, 0].scatter(self.MU_pos.cpu()[:, 0], self.MU_pos.cpu()[:, 1], c=Bk_deg.cpu(), marker=".", s=1)  # MUs
        ax[0, 0].scatter(BS_pos[0].cpu(), BS_pos[1].cpu(), c="red", marker="^", label="BS", s=1)  # BS
        ax[0, 0].set_xlabel("x")
        ax[0, 0].set_ylabel("y")
        ax[0, 0].set_title("BS-MU")
        ax[0, 0].set_aspect(aspect="equal")
        fig.colorbar(im_Bk, ax=ax[0, 0])

        # RIS-MU (K, N)
        Sk_deg = torch.mean(self.B_Sk.type(torch.float32), dim=1)  # degree of the blocking condition on RIS-MU
        im_Sk = ax[0, 1].scatter(self.MU_pos.cpu()[:, 0], self.MU_pos.cpu()[:, 1], c=Sk_deg.cpu(), marker=".", s=1)  # MUs
        ax[0, 1].scatter(BS_pos[0].cpu(), BS_pos[1].cpu(), c="red", marker="^", label="BS", s=1)  # BS
        ax[0, 1].set_xlabel("x")
        ax[0, 1].set_ylabel("y")
        ax[0, 1].set_title("RIS-MU")
        ax[0, 1].set_aspect(aspect="equal")
        fig.colorbar(im_Sk, ax=ax[0, 1])

        # BS-RIS (N, M)
        v, h = torch.meshgrid(
            torch.arange(self.ris_size[0] * d / 2, -self.ris_size[0] * d / 2, -d),
            torch.arange(-self.ris_size[1] * d / 2, self.ris_size[1] * d / 2, d),
            indexing="ij",
        )
        BS_deg = torch.mean(self.B_BS.type(torch.float32), dim=1)  # degree of the blocking condition on RIS-MU
        im_BS = ax[1, 0].scatter(h.flatten().cpu(), v.flatten().cpu(), c=BS_deg.cpu(), marker=".", s=1)
        ax[1, 0].set_title("BS-RIS")

        ax[1, 0].set_xlabel("horizontal")
        ax[1, 0].set_ylabel("vertical")
        ax[1, 0].set_aspect(aspect="equal")
        fig.colorbar(im_BS, ax=ax[1, 0])

        # RIS-MU (M, N)
        v, h = torch.meshgrid(
            torch.arange(self.ris_size[0] * d / 2, -self.ris_size[0] * self.d / 2, -d),
            torch.arange(-self.ris_size[1] * d / 2, self.ris_size[1] * self.d / 2, d),
            indexing="ij",
        )
        Sk_deg = torch.mean(self.B_Sk.type(torch.float32), dim=0)  # degree of the blocking condition on RIS-MU
        im_Sk = ax[1, 1].scatter(h.flatten().cpu(), v.flatten().cpu(), c=Sk_deg.cpu(), marker=".", s=1)
        ax[1, 1].set_title("RIS-MU")

        ax[1, 1].set_xlabel("horizontal")
        ax[1, 1].set_ylabel("vertical")
        ax[1, 1].set_aspect(aspect="equal")
        fig.colorbar(im_Sk, ax=ax[1, 1])

        fig.tight_layout()
        fig.savefig(f"{dir}/visi_cond_{self.scenario}.png", dpi=300)
        print(f'Saved: "{dir}/visi_cond_{self.scenario}.png"')

    def show(self, dir: str) -> None:
        """Plot the environment.
        Args:
            dir: The directory to dump the figure.
        """
        self.create()

        area = self.area
        BS_pos = self.BS_pos
        ris_pos = self.ris_pos

        fig = plt.figure()
        ax = plt.subplot(projection="3d")

        # Range of simulation
        ground_vertices = [
            list(
                [
                    [+area[0] / 2, +area[1] / 2, 0],
                    [-area[0] / 2, +area[1] / 2, 0],
                    [-area[0] / 2, -area[1] / 2, 0],
                    [+area[0] / 2, -area[1] / 2, 0],
                    [+area[0] / 2, +area[1] / 2, 0],
                ]
            )
        ]
        poly = Line3DCollection(ground_vertices, color="k", linewidths=1)
        ax.add_collection3d(poly)

        ax.scatter(ris_pos.view(-1, 3)[:, 0].cpu(), ris_pos.view(-1, 3)[:, 1].cpu(), ris_pos.view(-1, 3)[:, 2].cpu(), c="royalblue", marker="s", label="RIS")  # RIS

        for obst in self.obstacles:  # obstacles
            for poly in obst.polies():
                ax.add_collection3d(poly)

        ax.scatter(self.MU_pos.cpu()[:, 0], self.MU_pos.cpu()[:, 1], 0, c="limegreen", marker=2, label="MU")  # MUs

        ax.scatter(BS_pos[0].cpu(), BS_pos[1].cpu(), BS_pos[2].cpu(), c="red", marker="^", label="BS")  # BS

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(-area[0] / 2, area[0] / 2)
        ax.set_ylim(-area[1] / 2, area[1] / 2)
        ax.set_zlim(zmin=0)
        ax.set_aspect(aspect="equal")
        ax.legend()
        ax.grid(visible=False)

        fig.savefig(f"{dir}/preview_{self.scenario}.png", dpi=300)
        print(f'Saved: "{dir}/preview_{self.scenario}.png"')
