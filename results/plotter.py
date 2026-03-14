import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_data(filename: str, xlabel:str, data_type:str, benchmark:str):
    # Load the data from the CSV file
    full_path = os.path.join(os.getcwd(), 'results', filename)
    data = pd.read_csv(full_path, header=None, names=[xlabel,'Time','Bandwidth'])

    time_title = data_type + " - Compute Performance - " + benchmark
    bw_title = data_type + " - Memory Performance - " + benchmark


    # Plotting time:
    plt.figure(figsize=(10, 6))
    plt.plot(data[xlabel], data['Time'])
    plt.title(time_title)
    plt.xlabel(xlabel)
    plt.ylabel('Time (ms)')
    plt.grid(True)
    plt.savefig(full_path.replace('.csv', '_time.png'))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(data[xlabel], data['Bandwidth'])
    plt.title(bw_title)
    plt.xlabel(xlabel)
    plt.ylabel('Bandwidth (GB/s)')
    plt.grid(True)
    plt.savefig(full_path.replace('.csv', '_bw.png'))
    plt.show()

def benchmark_selector(name:str):
    if name == "varsize":
        return ("Array size", "Array size")
    elif name == "varj":
        return ("Multiple elements /thread","Elements/thread")
    elif name == "vark":
        return ("Workload (ops/thread)", "Operations/thread")

if __name__ == "__main__":
    print("Starting to plot data from CSV files...")
    pwd = os.getcwd()
    results_dir = os.path.join(pwd, 'results')
    print(os.listdir(results_dir))
    for filename in os.listdir(results_dir):
        if filename.endswith('.csv'):
            print(f"Plotting data from {filename}...")

            name_cut = filename.split('_')
            data_type = name_cut[0].capitalize()
            benchmark,xlabel = benchmark_selector(name_cut[2].replace(".csv",""))
            plot_data(filename, xlabel, data_type, benchmark)
    