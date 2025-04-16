import pandas as pd

# Load the uploaded CSV file
file_path = 'generate_RIS/all_phase_combinations/test00_small/all_phase_combinations_2-bits.csv'
# df = pd.read_csv(file_path)
df = pd.read_csv(file_path, nrows=5000)

# Display the first few rows of the dataframe to check the content
df.head()

# Define the relevant column ranges
element_columns = df.iloc[:, 0:2048]  # A to BZT
sinr_columns = df.iloc[:, 2048:2081]  # BZU to CAJ (SINR columns)
datarate_columns = df.iloc[:, 2081:2106]  # CAK to CAZ (Datarate columns)

# Calculate average SINR and Datarate
average_sinr = sinr_columns.mean(axis=1)
average_datarate = datarate_columns.mean(axis=1)

# Create x-axis for the combinations (use index)
x = range(len(df))

# Plot Average SINR
plt.figure(figsize=(10, 6))
plt.plot(x, average_sinr, label='Average SINR')
plt.xlabel('Combinations')
plt.ylabel('Average SINR')
plt.title('Average SINR vs Combinations')
plt.grid(True)
plt.show()

# Plot Average Datarate
plt.figure(figsize=(10, 6))
plt.plot(x, average_datarate, label='Average Datarate', color='orange')
plt.xlabel('Combinations')
plt.ylabel('Average Datarate')
plt.title('Average Datarate vs Combinations')
plt.grid(True)
plt.show()