import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_data(filename: str, xlabel:str, title:str):
    # Load the data from the CSV file
    full_path = os.path.join(os.getcwd(), 'results', filename)
    data = pd.read_csv(full_path, header=None, names=[xlabel,'Time','Bandwidth'])
    print(data)

    # Plotting time:
    plt.figure(figsize=(10, 6))
    plt.plot(data[xlabel], data['Time'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.savefig(full_path.replace('.csv', '_time.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data[xlabel], data['Bandwidth'])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True)
    plt.savefig(full_path.replace('.csv', '_bw.png'))
    plt.show()
    
if __name__ == "__main__":
    print("Starting to plot data from CSV files...")
    pwd = os.getcwd()
    results_dir = os.path.join(pwd, 'results')
    print(os.listdir(results_dir))
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            print(f"Plotting data from {filename}...")
            plot_data(filename, f"Size {filename.split('_')[2].split('.')[0]}", f"{filename.split('_')[0]} Performance vs {filename.split('_')[2].split('.')[0].capitalize()}")
    