import matplotlib.pyplot as plt
import datetime
import time
from matplotlib import font_manager, rc

now = time

# fixed broken Korean font
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def network_plot(filename, name):
    times = []
    rx_data = []
    tx_data = []
    with open(filename, 'r') as file:
        for line in file:
            if 'eth0' in line and not line.startswith('Average'):
                parts = line.split()
                time = datetime.datetime.strptime(parts[0] + ' ' + parts[1], '%I:%M:%S %p')
                rx = float(parts[4])
                tx = float(parts[5])
                times.append(time)
                rx_data.append(rx)
                tx_data.append(tx)
                
    start = times[0]
    seconds = [(t - start).total_seconds() for t in times]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(seconds, rx_data, label='rxkB/s')
    ax.plot(seconds, tx_data, label='txkB/s')
    
    ax.set_xlabel('Time(sec)')
    ax.set_ylabel('(kB/s)')
    ax.set_title('황서경 / 2021124195\n'+'Network usage\n'+now.strftime('%Y-%m-%d %H:%M:%S'))
    ax.legend()
    ax.grid()
    
    plt.savefig(name)
    plt.show()
    
    return  

def mem_data(filename):
    times = []
    mem_usage = []
    with open(filename, 'r') as file:
        next(file)
        next(file)
        next(file)
        for line in file:
            if 'AM' in line or 'PM' in line:
                parts = line.split()
                time = datetime.datetime.strptime(parts[0] + ' ' + parts[1], '%I:%M:%S %p')
                memused = float(parts[4])
                times.append(time)
                mem_usage.append(memused)
                
    return times, mem_usage

def disk_data(filename):
    times = []
    disk_usage = []
    with open(filename, 'r') as file:
        for line in file:
            if 'sda' in line and not line.startswith("Average:"):
                parts = line.split()
                time = datetime.datetime.strptime(parts[0] + ' ' + parts[1], '%I:%M:%S %p')
                util = float(parts[-1])
                times.append(time)
                disk_usage.append(util)
                
    return times, disk_usage

def cpu_data(filename):
    times = []
    cpu_usage = []
    with open(filename, 'r') as file:
        for line in file:
            if 'all' in line and not line.startswith('Average:'):
                parts = line.split()
                time = datetime.datetime.strptime(parts[0] + ' ' + parts[1], '%I:%M:%S %p')
                idle = float(parts[-1])
                times.append(time)
                cpu_usage.append(100 - idle)
                    
    return times, cpu_usage
    
def data_plot(x, y, ylabel, resource, name):        
    start = x[0]
    seconds = [(t - start).total_seconds() for t in x]
    
    plt.figure(figsize = (12, 6))
    plt.plot(seconds, y)
    
    plt.xlabel('Time(sec)')
    plt.ylabel(ylabel)
    plt.title('황서경 / 2021124195 \n'+resource+' usage\n'+now.strftime('%Y-%m-%d %H:%M:%S'))
    plt.legend()
    plt.grid()
    
    plt.savefig(name)
    plt.show()
    
    return

network_plot('network_wc.txt', 'Plot_networkWC.pdf')
network_plot('network_youtube.txt', 'Plot_networkYoutube.pdf')

x, y = mem_data('memory_wc.txt')
data_plot(x, y, '%memused(%)', 'Memory', 'Plot_memWC.pdf')

x, y = mem_data('memory_youtube.txt')
data_plot(x, y, '%memused(%)', 'Memory', 'Plot_memYoutube.pdf')

x, y = disk_data('disk_dat_wc.txt')
data_plot(x, y, '%util(%)', 'Disk', 'Plot_diskWC.pdf')

x, y = disk_data('disk_dat_youtube.txt')
data_plot(x, y, '%util(%)', 'Disk', 'Plot_diskYoutube.pdf')

x, y = cpu_data('cpu_wc.txt')
data_plot(x, y, '(100 - %idle)(%)', 'CPU', 'Plot_cpuWC.pdf')

x, y = cpu_data('cpu_youtube.txt')
data_plot(x, y, '(100 - %idle)(%)', 'CPU', 'Plot_cpuYoutube.pdf')
