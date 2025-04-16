from environment import Obstacle


def scenario_configs(scenario: str):
    MU_mode = "poisson"  # use Poisson point process to model users
    wavelength = 0.116

    if scenario == "test11":
        # environment
        area = (100, 200)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs1 = Obstacle(pos=(-10, -40), size=(15, 15), height=10, rotate=45)
        obs2 = Obstacle(pos=(-10, 40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs1, obs2, obs3]

        # MUs
        centers = [(25, 75)]
        std = 15
        K = 100
    elif scenario == "test10":
        # environment
        area = (100, 200)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs1 = Obstacle(pos=(-10, -40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs1, obs3]

        # MUs
        centers = [(25, 75)]
        std = 15
        K = 100
    elif scenario == "test01":
        # environment
        area = (100, 200)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs2 = Obstacle(pos=(-10, 40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs2, obs3]

        # MUs
        centers = [(25, 75)]
        std = 15
        K = 100
    elif scenario == "test00":
        # environment
        area = (100, 200)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(25, 75)]
        std = 15
        K = 100
    elif scenario == "test01_2_30":
        # environment
        area = (100, 300)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs2 = Obstacle(pos=(-10, 40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs2, obs3]

        # MUs
        centers = [(25, 90), (25, 60)]
        std = 15
        K = 100
    elif scenario == "test01_3_30":
        # environment
        area = (100, 300)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs2 = Obstacle(pos=(-10, 40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs2, obs3]

        # MUs
        centers = [(25, 105), (25, 75), (25, 45)]
        std = 15
        K = 100
    elif scenario == "test01_4_30":
        # environment
        area = (100, 300)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (256, 512)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs2 = Obstacle(pos=(-10, 40), size=(15, 15), height=10, rotate=45)
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs2, obs3]

        # MUs
        centers = [(25, 120), (25, 90), (25, 60), (25, 30)]
        std = 15
        K = 100
    elif scenario == "general RIS 32x32_obstacles rectangle 1":
        # environment
        area = (50, 50)

        # BS
        BS_pos = (3, 5, 0)

        # RIS
        ris_size = (16, 16)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(-10, -15)]
        std = 10
        K = 50
    elif scenario == "general RIS 32x32_obstacles rectangle 2":
        # environment
        area = (50, 50)

        # BS
        BS_pos = (3, 5, 0)

        # RIS
        ris_size = (16, 16)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(-10, 0)]
        std = 10
        K = 50
    elif scenario == "general RIS 32x32_obstacles rectangle 3":
        # environment
        area = (100, 200)

        # BS
        BS_pos = (25, -75, 0)

        # RIS
        ris_size = (32, 64)
        ris_norm = (1, 0, 0)
        ris_height = 20

        # Buildings
        obs3 = Obstacle(pos=(25, 0), size=(30, 30), height=10, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(25, 75)]
        std = 15
        K = 100
    elif scenario == "test00_small":
        # environment
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 64)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        # obs1 = Obstacle(pos=(-1.0, -4.0), size=(1.5, 1.5), height=1, rotate=45)
        # obs2 = Obstacle(pos=(-1.0, 4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs3 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 16
    elif scenario == "test01_small":
        # environment
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 64)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        # obs1 = Obstacle(pos=(-1.0, -4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs2 = Obstacle(pos=(-1.0, 4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs3 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs2, obs3]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 16
    elif scenario == "test10_small":
        # environment
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 64)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs1 = Obstacle(pos=(-1.0, -4.0), size=(1.5, 1.5), height=1, rotate=45)
        # obs2 = Obstacle(pos=(-1.0, 4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs3 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1, obs3]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 16
    elif scenario == "test11_small":
        # environment
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 64)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs1 = Obstacle(pos=(-1.0, -4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs2 = Obstacle(pos=(-1.0, 4.0), size=(1.5, 1.5), height=1, rotate=45)
        obs3 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1, obs2, obs3]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 16
    elif scenario == "debug":
        # scale = 350
        scale =  5

        # environment
        area = (10 * scale, 20 * scale)

        # BS
        BS_pos = (2.5 * scale, -7.5 * scale, 3)

        # RIS
        ris_size = (8, 8)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs3 = Obstacle(pos=(2.5 * scale, 0 * scale), size=(3.0 * scale, 3.0 * scale), height=9, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(2.5 * scale, 7.5 * scale)]
        std = 1 * scale
        K = 1
    else:
        print(f'Scenario "{scenario}" is not defined.')
        exit()

    M = 4
    d_ris_elem = wavelength / 2
    ris_center = (-area[0] / 2, 0, ris_height + d_ris_elem * ris_size[0] / 2)
    return wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode
