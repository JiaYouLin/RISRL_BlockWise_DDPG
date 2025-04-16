import torch
import os
from channel import Channel, Beamformer
from utils import scenario_configs
from math import pi, log10
# import math
# import einops

# import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, Union
from utils import gpu, scenario
import pandas as pd

import itertools
from matplotlib.ticker import FormatStrFormatter

if __name__ == '__main__':        
    # torch.set_printoptions(precision=12, threshold=10_000)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))    # Change working directory to script's location
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print('=====================Parameters=====================')
    print(f'Device: {device}')
    print(f'Scenario: {scenario}')
    print('====================================================\n')

    # Get scenario config from scenario_configs in utils.py
    wavelength, d_ris_elem, area, BS_pos, ris_size, ris_norm, ris_center, obstacles, centers, std, M, K, MU_mode = scenario_configs(scenario)

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
        MU_dist=MU_mode
    )
    beamformer = Beamformer(device=device)

    channel.config_scenario(
        centers_=centers,
        obstacles=obstacles,
        std_radius=std,
        scenario_name=scenario
    )
    channel.create()

    h_ris = ris_size[0]
    w_ris = ris_size[1]

    # ======================Test 0 (NO RIS, Z=0)======================
    print('==============Test 0 (NO RIS)==============')
    noris_n_timeslot = 1

    # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
    channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

    # element switch
    Z_noris = torch.zeros(noris_n_timeslot, h_ris * w_ris, dtype=torch.float32, device=device)
    
    Z_noris_complex = torch.polar(Z_noris, torch.zeros_like(Z_noris))

    H_noris = channel.get_joint_channel_coefficient(Z_noris_complex)
    W_noris = beamformer.MRT(H_noris)

    # SINR
    sinr_linear_noris = channel.SINR(H_noris, W_noris)
    sinr_db_noris = 10*torch.log10(sinr_linear_noris).cpu().numpy()  
    # show and format
    formatted_sinr_db_noris = [f'{x:.10f}' for x in sinr_db_noris.flatten()]
    print(f'No RIS SINR(dB): {formatted_sinr_db_noris}\n')
    sinr_noris_np = sinr_db_noris

    # Datarate
    if sinr_linear_noris is not None:
        # Calculate channel capacity based on Shannon's theorem
        # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
        channel_capacity_noris = torch.log2(1 + sinr_linear_noris)
        # Assume each symbol can carry 1 bit
        # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
        datarate_noris = channel_capacity_noris
    else:
        datarate_noris = 0
    print(f"No RIS Datarate: {datarate_noris}\n")
    datarate_noris_np = datarate_noris.cpu().numpy()

    # average SINR, Datarate and total Datarate
    avg_sinr_noris = np.mean(sinr_noris_np)
    avg_datarate_noris = np.mean(datarate_noris_np)
    total_datarate_noris = np.sum(datarate_noris_np)

    # Save each UE's SINR and Datarate to the dataframe
    df_noris = pd.DataFrame()
    for i in range(len(sinr_db_noris[0])):
        df_noris[f'UE{i+1}\'s_SINR'] = [sinr_db_noris[0][i]]
    for i in range(len(datarate_noris_np[0])):
        df_noris[f'UE{i+1}\'s_Datarate'] = [datarate_noris_np[0][i]]
    df_noris['Average_SINR'] = [avg_sinr_noris]
    df_noris['Average_Datarate'] = [avg_datarate_noris]
    df_noris['Total_Datarate'] = [total_datarate_noris]

    # Save to CSV
    csv_path_noris = os.path.abspath(f'./Scenario/{scenario}')
    if not os.path.isdir(csv_path_noris):
        os.makedirs(csv_path_noris)
    file_path_noris = os.path.join(csv_path_noris, 'no_ris_sinr_datarate.csv')
    df_noris.to_csv(file_path_noris, index=False)           # index=False means not writing the row index

    print(f'Test 0 NO RIS: SINR and Datarate without RIS saved to {csv_path_noris}.')

    #======================Test 7 (All phase combinations by bits)======================
    print('==============Test 7 (All phase combinations by bits)==============')    
    # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
    channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

    # Simulate {llcomb_n_timeslot} timeslots, e.g., 1000 timeslots, with multiple parallel environments
    allcomb_n_timeslot = 1                   

    Z_allcomb = torch.ones((allcomb_n_timeslot, h_ris*w_ris), device=device)        # blocking condition, a binary matrix
    # print(f'Z_allcomb: {Z_allcomb} \nshape: {Z_allcomb.shape}\n')

    # RIS phase shift bit count (2 bits (4), 4 bits (16), 6 bits (64), 8 bits (256))
    bits = 2
    # Sample the number of outputs
    sample = 50000
    # Define the phase selection list, generating 2^n phase values uniformly distributed within the [0, 2pi] interval
    phase_allcomb = np.linspace(0, 2*np.pi, 2**bits, endpoint=False).tolist()
    # print(f'phase_allcomb[:64]: {phase_allcomb[:64]}, len: {len(phase_allcomb)}\n')
    # Retrieve and calculate the actual number of RIS elements in the current scenario
    required_phase_count = ris_size[0] * ris_size[1]
    # print(f'required_phase_count: {required_phase_count}\n')

    # Set the total number of combinations to process, limited to the first sample combinations
    total_combinations = min(sample, len(phase_allcomb)**required_phase_count)
    print(f'Total combinations (limited to {sample}): {total_combinations}')
    # Generate the first sample combinations of RIS element phase shifts
    all_combinations = itertools.islice(itertools.product(phase_allcomb, repeat=required_phase_count), total_combinations)
    # Define the number of combinations processed in each batch (e.g., batch_size) and calculate the total number of batches
    batch_size = 1000
    num_batches = total_combinations // batch_size + (1 if total_combinations % batch_size != 0 else 0)

    # Set a maximum file size limit of 100MB for each CSV file
    max_file_size = 50 * 1024 * 1024  # 100 >> 50MB, 50 >> 25MB
    file_count = 1          # Document counter, used for file naming

    # Save to CSV
    csv_path_allcomb = f'./generate_RIS/all_phase_combinations/{scenario}'
    os.makedirs(csv_path_allcomb, exist_ok=True)
    filename_allcomb = f'all_phase_combinations_{bits}-bits_part{file_count}.csv'
    csv_path_allcomb_addfile = os.path.join(csv_path_allcomb, filename_allcomb)
    # Clear the old CSV file by opening it in 'w' mode, ensuring previous content is not appended during each execution
    with open(csv_path_allcomb_addfile, 'w') as f:
        pass
    current_file_size = 0
    # Set column name
    column_n = [f'Element{i+1}\'s_Phase' for i in range(required_phase_count)] + \
                [f'UE{i+1}\'s_SINR' for i in range(K)] + \
                [f'UE{i+1}\'s_Datarate' for i in range(K)] + \
                ['Average_SINR', 'Average_Datarate', 'Total_Datarate']
    # results = pd.DataFrame(columns=column_n)
    
    # For Plot
    sinr_values = []
    datarate_values = []

    # Start the progress bar, with total as total_combinations, reflecting the overall progress of sample combinations
    # Progress bar format: Percentage Progress || current processed combinations/total combinations [elapsed time (min:sec) <estimated time to complete (hr:min:sec), combinations processed per second]
    with tqdm(desc="Processing Phase Combinations", total=total_combinations, unit="combination") as pbar:
        
        # Initialize a flag to track if it's the first write (only write the header on the first write)
        first_write = True

        while True:
            # Extract the next batch of combinations
            batch_combinations = list(itertools.islice(all_combinations, batch_size))
            if not batch_combinations:          # Exit the loop if no combinations are left
                break
            
            # Start calculating the SINR and data rate for the current batch
            for combo in batch_combinations:
                # print(f'combo: {combo}\n')

                # Convert the current combination to a tensor and transfer it to CUDA
                theta_allcomb_radians = torch.tensor(combo, dtype=torch.float32).reshape(1, -1).to(device)
                # print(f'theta_alldir_radians: {theta_allcomb_radians}')

                # Convert radians to complex numbers
                theta_allcomb_complex = torch.polar(torch.ones_like(theta_allcomb_radians, dtype=torch.float32), theta_allcomb_radians).to(device)
                # print(f'theta_allcomb_complex: {theta_allcomb_complex}')

                Z_theta_allcomb = Z_allcomb * theta_allcomb_complex
                
                H_allcomb = channel.get_joint_channel_coefficient(Z_theta_allcomb)
                W_allcomb = beamformer.MRT(H_allcomb)
                
                # 計算SINR和Datarate
                sinr_linear_allcomb = channel.SINR(H_allcomb, W_allcomb)
                sinr_db_allcomb = 10*torch.log10(sinr_linear_allcomb).cpu().numpy()  
                # print(f"SINR(linear): {sinr_linear_allcomb}, shape: {sinr_linear_allcomb.shape} \n")
                # print(f"SINR(dB): {sinr_db_allcomb}, shape: {sinr_db_allcomb.shape}, type: {type(sinr_db_allcomb)} \n")

                if sinr_linear_allcomb is not None:
                    # Calculate channel capacity based on Shannon's theorem
                    # Assuming a Gaussian channel, the capacity is C = log2(1 + sinr)
                    channel_capacity_allcomb = torch.log2(1 + sinr_linear_allcomb)
                    # Assume each symbol can carry 1 bit
                    # The data rate is the channel capacity multiplied by the symbol rate (assumed to be 1 here)
                    datarate_allcomb = channel_capacity_allcomb
                else:
                    datarate_allcomb = 0
                # print(f"Datarate: {datarate}, shape: {datarate.shape}, type: {type(datarate)} \n")

                # Calculate the Avg. SINR, Total Datarate, and Avg. Datarate for UE 1~16.
                avg_sinr_allcomb = np.mean(sinr_db_allcomb[:16])
                avg_datarate_allcomb = datarate_allcomb[:16].mean().cpu().item()
                total_datarate_allcomb = datarate_allcomb[:16].sum().cpu().item()

                # Ensure combo is correctly saved and separates SINR and datarate
                row = list(combo) + list(sinr_db_allcomb.flatten()) + list(datarate_allcomb.flatten().cpu().numpy()) + [avg_sinr_allcomb, avg_datarate_allcomb, total_datarate_allcomb]
                # Convert row to DataFrame format
                allcomb_row_df = pd.DataFrame([row], columns=column_n)


                # Check file size and switch files if necessary
                current_file_size += allcomb_row_df.memory_usage(deep=True).sum()
                if current_file_size > max_file_size:
                    file_count += 1
                    filename_allcomb = f'all_phase_combinations_{bits}-bits_part{file_count}.csv'
                    csv_path_allcomb_addfile = os.path.join(csv_path_allcomb, filename_allcomb)
                    with open(csv_path_allcomb_addfile, 'w') as f:
                        pass
                    current_file_size = 0
                    first_write = True  # A new file needs to write headers

                # Save data to the current file (append mode)
                allcomb_row_df.to_csv(csv_path_allcomb_addfile, mode='a', header=first_write, index=False)

                # For Plot
                sinr_values.append(sinr_db_allcomb.flatten())
                datarate_values.append(datarate_allcomb.flatten())
                
                # Keep the header only during the first write, and skip writing the header in subsequent writes
                first_write = False

            # Update the progress bar
            pbar.update(len(batch_combinations))

    print(f'Test 7 allcomb: all_phase_combinations saved to {csv_path_allcomb_addfile}.')

    # ========================Plot: Plot Configuration and Data Sampling========================
    # Calculate number of data points; auto-adjust figure width: min 20, max 150 based on data points
    num_points = len(sinr_values)
    fig_width = max(20, min(150, num_points / 100))

    # Dynamically adjust figure height based on SINR range, constrained between 5 and 12
    sinr_range = max([x.mean() for x in sinr_values]) - min([x.mean() for x in sinr_values])
    fig_height = max(5, min(12, sinr_range * 5))

    # Sample data for display if too large, with a maximum of 5000 points shown
    max_points = 5000
    if num_points > max_points:
        indices = np.linspace(0, num_points - 1, max_points).astype(int)
        sinr_values_sampled = [sinr_values[i] for i in indices]
        datarate_values_sampled = [datarate_values[i] for i in indices]
    else:
        sinr_values_sampled = sinr_values
        datarate_values_sampled = datarate_values

    # Auto-adjust X-axis ticks: show a tick every 10% of the data, at least one tick
    xticks_interval = max(1, len(sinr_values_sampled) // 10)

    # ============Plot: SINR============
    sinr_values_sampled = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in sinr_values_sampled]
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    plt.plot(
        range(1, len(sinr_values_sampled) + 1), 
        [x.mean() for x in sinr_values_sampled],                # Conversion is no longer needed here as it's already NumPy
        label='SINR (dB)', 
        color='brown'
    )

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.xlabel('Phase Combination', fontsize=12)
    plt.ylabel('SINR (dB)', fontsize=12)
    plt.title('Each phase combination\'s SINR', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(sinr_values_sampled) + 1, xticks_interval), fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    sinr_fig_name = f'SINR_{bits}-bits.png'
    allcomb_sinr_plot_path = os.path.join(csv_path_allcomb, sinr_fig_name)
    plt.savefig(allcomb_sinr_plot_path)

    print(f'Test 7 allcomb: SINR plot saved to {allcomb_sinr_plot_path}.')

    # ============Plot: Datarate============
    datarate_values_sampled = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in datarate_values_sampled]
    plt.figure(figsize=(fig_width, fig_height), dpi=300)
    plt.plot(
        range(1, len(datarate_values_sampled) + 1), 
        [x.mean() for x in datarate_values_sampled],                # Conversion is no longer needed here as it's already NumPy
        label='Datarate', 
        color='blue'
    )

    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.xlabel('Phase Combination', fontsize=12)
    plt.ylabel('Datarate', fontsize=12)
    plt.title('Each phase combination\'s Datarate', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, len(datarate_values_sampled) + 1, xticks_interval), fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    datarate_fig_name = f'Datarate_{bits}-bits.png'
    allcomb_datarate_plot_path = os.path.join(csv_path_allcomb, datarate_fig_name)
    plt.savefig(allcomb_datarate_plot_path)

    print(f'Test 7 allcomb: Datarate plot saved to {allcomb_datarate_plot_path}.')

    #======================test 8 (random phase shift)======================
    print('==============Test 8 (random phase shift)==============')

    # Simulate {n_env} timeslots, e.g., 1000 timeslots, with multiple parallel environments
    n_env = 1000

    # Save CSV path: Ensure the directory exists
    random_csv_path = f'./generate_RIS/{scenario}'
    os.makedirs(random_csv_path, exist_ok=True)

    all_data = []  # Store all timeslot data for single CSV export at the end

    for i in range(n_env):  # Simulate each timeslot
        
        # Initialize by updating the channel (UE positions unchanged), call "channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)." To change UE positions, call "channel.update_environment()."
        channel.get_channel_coefficient()       # get function
        # init_channel = channel.get_channel_coefficient()          # test
        # print(f"Initial Channel Coefficient: {init_channel}")
        channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)
        # update_channel = channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)          # test
        # print(f"Initial Channel Coefficient: {update_channel}")
        # after_update_channel = channel.get_channel_coefficient()          # test
        # print(f"Initial Channel Coefficient: {after_update_channel}")
        
        # if i == 3:        # test
        #     exit()

        Z_random = torch.ones((1, h_ris*w_ris), device=device)              # blocking condition, a binary matrix
        # print(f'Z_random: {Z_random} \nshape: {Z_random.shape}\n')

        # Generate random values in radians between 0 and 2pi
        theta_random = torch.rand_like(Z_random, device=device) * 2*pi
        # print(f'theta_random: {theta_random}')
        # show and format
        formatted_theta_random = [f'{x:.10f}' for x in theta_random.flatten()]
        # print(f'formatted_theta_random[:60]: {formatted_theta_random[:60]}...\n')

        # Define discrete phase states (in radians)
        phases_discrete = torch.tensor([0, pi/2, pi, 3*pi/2], device=device)
        # print(f'phases_discrete: {phases_discrete}\n')

        # Create a tensor with the same shape as theta_random to store discrete phases
        theta_random_radians = torch.zeros_like(theta_random)
        # Convert continuous phases to discrete phases
        # for i in range(theta_random.size(0)):  # Loop through each environment
        for j in range(theta_random.size(1)):  # Loop through each RIS elements
            # Calculate the distance to each discrete phase.
            distances = torch.abs(theta_random[0, j] - phases_discrete)
            # distances = torch.abs(theta_random[i, j] - phases_discrete)
            # Find the index of the nearest discrete phase.
            min_index = torch.argmin(distances)
            # Assign the nearest discrete phase to theta_random_radians
            theta_random_radians[0, j] = phases_discrete[min_index]
            # theta_random_radians[i, j] = phases_discrete[min_index]
        # print(f'theta_random_radians: {theta_random_radians}\n')
        # print(f'theta_random_radians shape: {theta_random_radians.shape}\n')

        # show and format
        formatted_theta_random_radians = [f'{x:.10f}' for x in theta_random_radians.flatten()]
        print(f'formatted_theta_random_radians[:60]: {formatted_theta_random_radians[:60]}...\n')

        # Convert radians to complex numbsers
        theta_random_complex = torch.polar(torch.ones_like(Z_random, dtype=torch.float32), theta_random_radians).to(device)
        # print(f'theta_random_complex: {theta_random_complex}\n')

        # element switch
        Z_theta_random = Z_random * theta_random_complex
        
        H_random = channel.get_joint_channel_coefficient(Z_theta_random) 
        W_random = beamformer.MRT(H_random)

        # SINR
        sinr_linear_random = channel.SINR(H_random, W_random)
        sinr_db_random = 10*torch.log10(sinr_linear_random).cpu().numpy()  
        # print(f'Random SINR(linear): {sinr_linear_random}')
        # print(f'Random SINR(dB): {sinr_db_random}\n')
        # print(f'len(Random SINR(dB)): {len(sinr_db_random)}\n')
        # show and format
        formatted_sinr_db_random = [f'{x:.10f}' for x in sinr_db_random.flatten()]
        print(f'Random RIS element SINR(dB): {formatted_sinr_db_random}\n')

        # ----------------------------------------------------------------------
        # print(f'Random Z\'s shape: {Z_random.shape}')                             #([1, 1024])
        # print(f'Random phase\'s shape: {theta_random_radians.shape}')             # ([1, 1024])
        # print(f'Random SINR(dB)\'s shape: {sinr_db_random.shape}')                # ([n_env, 1024])
        
        # Ensure all are tensors, convert SINR to a tensor
        sinr_db_random_tensor = torch.from_numpy(sinr_db_random).to(theta_random_radians.device)
        # print(f'sinr_db_random_tensor: {sinr_db_random_tensor} \nshape: {sinr_db_random_tensor.shape}\n')

        # Concatenate phase and SINR along dim=1.
        Phase_state = torch.cat((theta_random_radians, sinr_db_random_tensor), dim=1)
        # print(f'Phase_state: {Phase_state} \nshape: {Phase_state.shape}\n')       # ([1, 1025])
        """
        我是模擬1000次, 看是要先存起來還是每次呼叫, 但要有一個計數的, 這樣才知道丟了多少timeslots進去
        應該是每個element要是一筆資料, 有1024個element所以會有1024筆資料, 每一筆資料後面會再加上1個sinr

        1000 timeslot是再外層的資料夾, 所以是在timeslot 1時的這個csv會有250車
        因為尚桓的是每一台車是一筆資料, 有250台車所以會有250筆資料, 每一筆資料後面會再加上1個sinr
        """

        # ----------------------Save the Phase and UE of each timeslot to a CSV----------------------    
        
        Phase_state_np = Phase_state.cpu().numpy()      # Convert tensor to NumPy and move to CPU
        
        num_elements = h_ris * w_ris
        num_ues = sinr_db_random_tensor.shape[1]
        column_names = [f'Element{i+1}' for i in range(num_elements)] + \
                        [f'UE{i+1}\'s_SINR' for i in range(num_ues)]

        # Save each timeslot individually with and without headers
        # Save without headers
        row_data = Phase_state_np[0]
        random_df = pd.DataFrame([row_data])
        eachtime_csv_path = os.path.join(random_csv_path, f'RIS_element_{h_ris*w_ris}_Phase_timeslot{i+1}.csv')
        random_df.to_csv(eachtime_csv_path, index=False, header=False)      # Save individual CSV file with numeric values only
        # print(f'Saved timeslot {i+1} to {eachtime_csv_path}.')

        # Save with headers
        random_df_marked = pd.DataFrame([row_data], columns=column_names)
        eachtime_csv_path_marked = os.path.join(random_csv_path, f'RIS_element_{h_ris*w_ris}_Phase_timeslot{i+1}_marked.csv')
        random_df_marked.to_csv(eachtime_csv_path_marked, index=True, header=True)      # Save individual CSV file with column names
        # print(f'Saved timeslot {i+1} to {eachtime_csv_path_marked}.')

        # Collect data for saving all timeslots to a single file later
        all_data.append(row_data)

        # Update UE position for each timeslot
        channel.update_environment()

    # # Save each row as a separate CSV file, with each timeslot stored individually.
    # for i in range(Phase_state_np.shape[0]):
    #     row_data = Phase_state_np[i, :]

    #     random_df = pd.DataFrame([row_data])
    #     eachtime_csv_path_temp = os.path.join(random_csv_path, f'RIS_element_{h_ris*w_ris}')
    #     eachtime_csv_path = f'{eachtime_csv_path_temp}_Phase_timeslot{i+1}.csv'
    #     random_df.to_csv(eachtime_csv_path, index=False, header=False)     # Save individual CSV file with numeric values only
    #     # print(f'Saved timeslot {i+1} to {eachtime_csv_path}.')

    #     random_df_marked = pd.DataFrame([row_data], columns=column_names)
    #     eachtime_csv_path_marked_temp = os.path.join(random_csv_path, f'RIS_element_{h_ris*w_ris}')
    #     eachtime_csv_path_marked = f'{eachtime_csv_path_marked_temp}_Phase_timeslot{i+1}_marked.csv'
    #     random_df_marked.to_csv(eachtime_csv_path_marked, index=True, header=True)     # Save individual CSV file with column names
    #     # print(f'Saved timeslot {i+1} to {eachtime_csv_path_marked}.')

    # ---------Save all timeslot data into a single CSV file---------
    # all_data = []
    index_names = [f'timeslot_{i+1}' for i in range(n_env)]

    alltime_df_all = pd.DataFrame(all_data, columns=column_names, index=index_names)
    alltime_csv_path = os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots_marked.csv')
    alltime_df_all.to_csv(alltime_csv_path, index=True, header=True)

    # for i in range(Phase_state_np.shape[0]):
    #     row_data = Phase_state_np[i, :]
    #     all_data.append(row_data)

    # alltime_df_all = pd.DataFrame(all_data, columns=column_names)
    # alltime_df_all.index = index_names

    # alltime_csv_path = os.path.join(random_csv_path, 'RIS_element_Phase_all_timeslots_marked.csv')
    # alltime_df_all.to_csv(alltime_csv_path , index=True, header=True)

    print(f'Test 8 Random: RIS_element_Phase for all timeslots saved to {alltime_csv_path }.')

    # ----------------------Draw scenario diagram----------------------
    print('====================================================')
    # Ensure the result directory exists; if not, create it
    dir = os.path.abspath(f'./Scenario/{scenario}')
    if not os.path.isdir(dir):
        os.makedirs(dir)
    # Plot and save the blockage conditions and overview of the scenario
    channel.plot_block_cond(dir=dir)  # Plot block condition
    channel.show(dir=dir)  # plot the scenario

    # ===================Test 9: Fixed Phase, Variable UE Position==================
    print("==== Test: Fixed Phase with Variable UE Positions ====")
    # Initialize a random discrete fixed phase configuration
    discrete_phases = torch.tensor([0, pi/2, pi, 3*pi/2], device=device)
    indices = torch.randint(0, len(discrete_phases), (1, h_ris * w_ris), device=device)
    fixed_phase = discrete_phases[indices]
    fixed_phase_complex = torch.polar(torch.ones_like(fixed_phase, dtype=torch.float32), fixed_phase).to(device)

    print(f"Fixed random discrete phase (radians): {fixed_phase.flatten()[:10]}...") 

    # Run multiple UE environment configurations with fixed random discrete RIS phase
    num_env_variations = 50  # Set the number of different test environments
    sinr_results = []
    phase_results = []

    # Save CSV path: Ensure the directory exists
    random_csv_path = f'./generate_RIS/{scenario}'
    os.makedirs(random_csv_path, exist_ok=True)

    for env_idx in range(num_env_variations):
        # Update UE positions while keeping the RIS phase fixed
        channel.update_environment()
        channel.update_channel(alpha_los=2, alpha_nlos=4, kapa=10, time_corrcoef=0)

        # Calculate SINR for the new UE environment with fixed random phases
        H_fixed = channel.get_joint_channel_coefficient(fixed_phase_complex)
        W_fixed = beamformer.MRT(H_fixed)
        sinr_linear_fixed = channel.SINR(H_fixed, W_fixed)
        sinr_db_fixed = 10 * torch.log10(sinr_linear_fixed).cpu().numpy()

        # Save the Phase and SINR results for each iteration
        sinr_results.append(sinr_db_fixed.flatten())
        phase_results.append(fixed_phase.cpu().numpy().flatten())

        print(f"Environment {env_idx + 1}/{num_env_variations} - Fixed Random Discrete Phase SINR (dB): {sinr_db_fixed.flatten()}")

    # Calculate the average value
    average_sinr = np.mean(sinr_results, axis=0)
    print(f"\nAverage SINR over {num_env_variations} environments with fixed random discrete phase: {average_sinr}")

    # Create DataFrames for phase and SINR results
    phase_df = pd.DataFrame(phase_results, columns=[f'Element_Phase_{i + 1}' for i in range(h_ris * w_ris)])
    sinr_df = pd.DataFrame(sinr_results, columns=[f'UE_{i + 1}_SINR' for i in range(K)])

    # Save to CSV
    results_df = pd.concat([phase_df, sinr_df], axis=1)
    fixed_phase_sinr_csv_path = os.path.join(random_csv_path, 'TestEnv_Fixed_random_discrete_phase_sinr_across_envs.csv')
    results_df.to_csv(fixed_phase_sinr_csv_path, index_label="Env_Index")

    print(f"Fixed random discrete phase SINR results saved to {fixed_phase_sinr_csv_path}")

    # # ======================Draw test scenario diagram.=========================
    # print('====================================================')
    # # Ensure the result directory exists; if not, create it
    # dir = os.path.abspath(f'./Scenario/test/{scenario}')
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # # Plot and save the blockage conditions and overview of the scenario.
    # channel.plot_block_cond(dir=dir)  # Plot block condition
    # channel.show(dir=dir)  # plot the scenario
