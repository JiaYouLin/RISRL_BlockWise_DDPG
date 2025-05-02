"""
現在BS的方向只能往-x方向打, BS現在是array antenna, 它的天線數量是M變數去調整, 如果M設16的話就是4*4
也因為方向性的關係, 在BS後面的UE都會收不到訊號
Tips: 如果要不考慮BS到UE間的path的話, 就把UE放到BS後方, 就收不到了, 可以當作Debug用
在確定這樣的環境配置可以調整好後,就把這條path放回BS前面就行了

環境配置大小或位置是否合理, 應該可以先假設室內情境(10*20), 先設小一點
RIS element應該就用32*32
Tips: RIS elements如果夠大的話, 打得Beam也會夠細, 應該就會打得更準, 但前提是要夠大的話

就用RIS尺寸去猜房間多大, 應該會比較合理
尺寸轉換真實大小的計算方式:
    如果波長是0.116公尺, 因為element spacing(天線元素之間的距離)是半個波長,
    元素間距即: 0.116/2=0.058公尺
    假設一個RIS有32*32的天線陣列表示有32行、32列的天線元件, 
    長和寬的單位是公尺, 但是因每個元素之間的距離為0.058公尺
    這樣的天線陣列的實際物理尺寸之總長度為32*0.058=1.856公尺, 寬度同理,
    因此, 這個32*32天線陣列的實際物理尺寸大約是1.856x1.856平方公尺,
    這是現實世界的物理尺寸大小,這個大小反映的是天線元件在空間中的排列布局
>> 應該只能用這種方式判斷空間物體大小的合理性

area=(10, 20)  # 單位是公尺, 即10*20平方公尺
"""

from environment import Obstacle

gpu = '0'
# scenario = "RIS16x16_UE50_2_square50_NoBlocked"           # -14~-10 dB, 正方, UE在LoS上, 障礙物無遮擋
# scenario = "RIS32x32_UE50_1_square50_FullyBlocked"        # -16 dB, 正方, UE在NLoS上, 障礙物完全遮擋
# scenario = "RIS32x32_UE10_1_square50_FullyBlocked"        # -9 dB, 正方, UE在NLoS上, 障礙物完全遮擋
# scenario = "RIS32x32_UE5_1_square50_FullyBlocked"        # -6 dB, 正方, UE在NLoS上, 障礙物完全遮擋

# scenario = 'RIS32x32_UE16_1_rectangle1020_FullyBlocked'     # 實作

# scenario = "RIS16x16_UE1_1_square50_HalfBlocked"          # Xuan-Yi
# scenario = "RIS32x32_UE1_1_rectangle110200_FullyBlocked"  # -16 dB, 長方, UE在NLoS上, 以障礙物為中心對稱完全遮擋
# scenario = "RIS32x32_UE1_2_square50_HalfBlocked"          # 29 dB, 正方, UE在NLoS上, 障礙物一半遮擋

# scenario = "RIS32x32_UE1_3_rectangle1020_FullyBlocked"      # 實作
# scenario = "RIS2x2_UE1_1_rectangle1020_FullyBlocked"      # 實作
# scenario = "RIS2x2_UE1_1_rectangle50100_FullyBlocked"      # 實作
# scenario = "debug_with_obs_ue1"      # 實作
# scenario = "debug_with_obs_ue4"      # 實作
scenario = "debug_with_obs_32_32_ue4"
# scenario = "debug_no_obs"      # 實作

# scenario = "test1"

# ==========test==========

# scenario = "RIS2x2_UE1_test_0_center_left_square50_FullyBlocked_smallarea"

# scenario = "RIS2x2_UE1_test_1_center_left_square50_FullyBlocked"         # 24 dB, 正方, UE在NLoS上, 障礙物完全遮擋
# scenario = "RIS2x2_UE1_test_2_center_left_square50_NoBlocked"         # 27 dB, 正方, UE在LoS上, 障礙物無遮擋
# scenario = "RIS2x2_UE1_test_3_center_left_square50_LargeBlocked"         # 24 dB, 正方, UE在LoS上, 障礙物無遮擋

# scenario = "RIS2x2_UE1_test_4_top_left_square50_FullyBlocked"         # 29 dB, 正方, UE在NLoS上, 障礙物完全遮擋
# scenario = "RIS2x2_UE1_test_5_top_left_square50_NoBlocked"         # 32 dB, 正方, UE在LoS上, 障礙物無遮擋
# scenario = "RIS2x2_UE1_test_6_top_left_square50_LargeBlocked"         # 29 dB, 正方, UE在LoS上, 障礙物無遮擋

# scenario = "RIS2x2_UE1_test_7_bottom_left_square50_FullyBlocked"         # 30 dB, 正方, UE在NLoS上, 障礙物完全遮擋
# scenario = "RIS2x2_UE1_test_8_bottom_left_square50_NoBlocked"         # 30 dB, 正方, UE在LoS上, 障礙物無遮擋
# scenario = "RIS2x2_UE1_test_9_bottom_left_square50_LargeBlocked"         # 30 dB, 正方, UE在LoS上, 障礙物無遮擋

# scenario = "RIS4x4_UE1_test_1_center_left_square50_FullyBlocked"         # 24 dB, 正方, UE在NLoS上, 障礙物完全遮擋

# scenario = "RIS3x2_UE4_test_1_center_left_square50_FullyBlocked"

# scenario = "RIS32x32_UE1_test_1_center_left__rectangle1020_FullyBlocked"

# scenario = 'test00_small_ue1'
# scenario = 'test00_small_ue16'

def scenario_configs(scenario: str):
    MU_mode = 'poisson' # use Poisson point process to model users
    wavelength = 0.116

    if scenario == 'RIS32x32_UE1_test_1_center_left__rectangle1020_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs1 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 1
    elif scenario == 'RIS3x2_UE4_test_1_center_left_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (-10, 10, 5)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (3, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(-5, 0), size=(20, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(10, -10)]
        std = 25
        K = 4
    elif scenario == 'RIS2x2_UE1_test_0_center_left_square50_FullyBlocked_smallarea':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (5, 5)

        # BS
        BS_pos = (0, 2, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (1, 1)
        ris_norm = (1, 0, 0)
        ris_height = 5

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(1, 1), height=3)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -2)]
        # centers = [(0, -15)]
        std = 0.001
        K = 1
    elif scenario == 'RIS2x2_UE1_test_1_center_left_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (-10, 10, 5)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(-5, 0), size=(20, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(10, -10)]
        # centers = [(0, -15)]
        std = 1
        K = 1
    elif scenario == 'RIS2x2_UE1_test_2_center_left_square50_NoBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(20, 0), size=(5, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_3_center_left_square50_LargeBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(30, 30), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_4_top_left_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 0, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, -10), size=(17, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -30)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_5_top_left_square50_NoBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 0, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(20, 0), size=(5, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -30)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_6_top_left_square50_LargeBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 0, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, -15), size=(20, 15), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -30)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_7_bottom_left_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, 10), size=(17, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, 0)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_8_bottom_left_square50_NoBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(-20, 20), size=(5, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, 0)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS2x2_UE1_test_9_bottom_left_square50_LargeBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 10)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (2, 2)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, 10), size=(25, 15), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, 0)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS4x4_UE1_test_1_center_left_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 0)
        # BS_pos = (0, 5, 0)

        # RIS
        ris_size = (4, 4)
        ris_norm = (1, 0, 0)
        ris_height = 15

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(17, 5), height=10)
        # obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        # centers = [(0, -15)]
        std = 10
        K = 1
    elif scenario == 'RIS16x16_UE1_1_square50_HalfBlocked':       # Xuan-yi
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
        K = 1
    elif scenario == 'RIS16x16_UE50_2_square50_NoBlocked':       # Xuan-yi
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
    elif scenario == 'RIS32x32_UE50_1_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 0)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(17, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        std = 10
        K = 50
    elif scenario == 'RIS32x32_UE10_1_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 0)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(17, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        std = 10
        K = 10
    elif scenario == 'RIS32x32_UE5_1_square50_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        area = (50, 50)

        # BS
        BS_pos = (0, 20, 0)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, 0), size=(17, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(0, -20)]
        std = 10
        K = 5
    elif scenario == 'RIS32x32_UE1_1_rectangle110200_FullyBlocked':       # Xuan-yi
        # environment
        area = (110, 200)

        # BS
        BS_pos = (-30, 30, 60)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(-30, 0), size=(5, 5), height=100)
        obstacles = [obs1]

        # MUs
        centers = [(-30, -30)]
        std = 5
        K = 1
    elif scenario == 'RIS32x32_UE1_2_square50_HalfBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        area = (50, 50)

        # BS
        BS_pos = (3, 5, 0)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 30

        # Buildings
        obs1 = Obstacle(pos=(0, -7), size=(7, 5), height=10)
        obstacles = [obs1]

        # MUs
        centers = [(-10, -15)]
        std = 10
        K = 1
    elif scenario == 'test':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        # area = (0.625, 1.25)            # (0.625, 1.25)
        # area = (5, 10)
        area = (10, 20)

        # BS, red
        # BS_pos = (0, 0.45, 0.1)         # (0, 0.25, 0.1)
        # BS_pos = (1.25, -3.75, 0.1)
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (2, 2)               # (2, 2)
        ris_norm = (1, 0, 0)            # (1, 0, 0)
        # ris_height = 0.25               # 0.25
        ris_height = 1.7

        # Buildings
        # obs1 = Obstacle(pos=(0, 0), size=(0.4, 0.4), height=0.3, rotate=45)     # pos=(0, 0), size=(0.052, 0.052), height=0.2, rotate=45
        # obs1 = Obstacle(pos=(1.25, 0), size=(1.5, 1.5), height=0.5, rotate=45)
        obs1 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1]

        # MUs, green
        # centers = [(0.52, 0.48)]         # (0.5, 0.25)
        # centers = [(0.52, 0)]         # (0.5, 0.25)
        # centers = [(1.85, 4)]
        centers = [(2.5, 7.5)]
        std = 1
        K = 1
    elif scenario == 'test1':
        scale = 5

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
    elif scenario == "debug_with_obs_ue1":
        scale = 5

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
    elif scenario == "debug_with_obs_ue4":
        scale = 5

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
        K = 4
    elif scenario == "debug_with_obs_32_32_ue4":
        scale = 5

        # environment
        area = (10 * scale, 20 * scale)

        # BS
        BS_pos = (2.5 * scale, -7.5 * scale, 3)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs3 = Obstacle(pos=(2.5 * scale, 0 * scale), size=(3.0 * scale, 3.0 * scale), height=9, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(2.5 * scale, 7.5 * scale)]
        std = 5 * scale
        K = 4
    elif scenario == "debug_no_obs":
        scale = 5

        # environment
        area = (10 * scale, 20 * scale)

        # BS
        BS_pos = (2.5 * scale, -7.5 * scale, 3)

        # RIS
        ris_size = (8, 8)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs3 = Obstacle(pos=(5 * scale, 0 * scale), size=(3.0 * scale, 3.0 * scale), height=9, rotate=45)
        obstacles = [obs3]

        # MUs
        centers = [(2.5 * scale, 7.5 * scale)]
        std = 1 * scale
        K = 1
    elif scenario == "RIS2x2_UE1_1_rectangle50100_FullyBlocked":
        scale = 5

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
    elif scenario == 'RIS2x2_UE1_1_rectangle1020_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        # area = (0.625, 1.25)            # (0.625, 1.25)
        # area = (5, 10)
        area = (10, 20)

        # BS, red
        # BS_pos = (0, 0.45, 0.1)         # (0, 0.25, 0.1)
        # BS_pos = (1.25, -3.75, 0.1)
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (2, 2)               # (2, 2)
        ris_norm = (1, 0, 0)            # (1, 0, 0)
        # ris_height = 0.25               # 0.25
        ris_height = 1.7

        # Buildings
        # obs1 = Obstacle(pos=(0, 0), size=(0.4, 0.4), height=0.3, rotate=45)     # pos=(0, 0), size=(0.052, 0.052), height=0.2, rotate=45
        # obs1 = Obstacle(pos=(1.25, 0), size=(1.5, 1.5), height=0.5, rotate=45)
        obs1 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1]

        # MUs, green
        # centers = [(0.52, 0.48)]         # (0.5, 0.25)
        # centers = [(0.52, 0)]         # (0.5, 0.25)
        # centers = [(1.85, 4)]
        centers = [(2.5, 7.5)]
        std = 1
        K = 1
    elif scenario == 'RIS32x32_UE1_3_rectangle1020_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs1 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1
        K = 1
    elif scenario == 'RIS32x32_UE16_1_rectangle1020_FullyBlocked':
        # environment
        # assert BS_pos[1] > (-area[1] / 2) and BS_pos[1] < (area[1] / 2)
        # >> (area 的 y 值 / 2) > BS_pos 的 y 值 > (-area 的 y 值 / 2)
        
        area = (10, 20)

        # BS
        BS_pos = (2.5, -7.5, 0.1)

        # RIS
        ris_size = (32, 32)
        ris_norm = (1, 0, 0)
        ris_height = 1.7

        # Buildings
        obs1 = Obstacle(pos=(2.5, 0), size=(3.0, 3.0), height=1, rotate=45)
        obstacles = [obs1]

        # MUs
        centers = [(2.5, 7.5)]
        std = 1.5
        K = 16
    elif scenario == 'test00_small_ue1':            # Xuan-Yi: genData_small version copy change ue=1
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
        std = 1
        K = 1
    elif scenario == 'test00_small_ue16':            # Xuan-Yi: genData_small version
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
    elif scenario == 'test01_small':            # Xuan-Yi: genData_small version
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
    elif scenario == 'test10_small':            # Xuan-Yi: genData_small version
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
    elif scenario == 'test11_small':            # Xuan-Yi: genData_small version
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
    
    else:
        print(f'Scenario \"{scenario}\" is not defined.')
        exit()

    # M = 1
    # d = wavelength / 2
    # ris_center = (-area[0] / 2, 0, ris_height + d * ris_size[0] / 2)
    # return wavelength, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode

    M = 4       # array antenna, 16就是4*4, 天線變大, beam會打得更細, 有可能會打得更準
    d_ris_elem = wavelength / 2
    ris_center = (-area[0] / 2, 0, ris_height + d_ris_elem * ris_size[0] / 2)
    return wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode, scale, ris_height, obs3