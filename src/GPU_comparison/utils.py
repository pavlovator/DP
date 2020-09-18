import subprocess

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

from matplotlib import pyplot as plt


def bar_plot(ax, data, colors=None, total_width=0.8, single_width=1, legend=True, ticks=None, title=""):
    """Draws a bar plot with multiple bars per data point.

    Parameters
    ----------
    ax : matplotlib.pyplot.axis
        The axis we want to draw our plot on.

    data: dictionary
        A dictionary containing the data we want to plot. Keys are the names of the
        data, the items is a list of the values.

        Example:
        data = {
            "x":[1,2,3],
            "y":[1,2,3],
            "z":[1,2,3],
        }

    colors : array-like, optional
        A list of colors which are used for the bars. If None, the colors
        will be the standard matplotlib color cyle. (default: None)

    total_width : float, optional, default: 0.8
        The width of a bar group. 0.8 means that 80% of the x-axis is covered
        by bars and 20% will be spaces between the bars.

    single_width: float, optional, default: 1
        The relative width of a single bar within a group. 1 means the bars
        will touch eachother within a group, values less than 1 will make
        these bars thinner.

    legend: bool, optional, default: True
        If this is set to true, a legend will be added to the axis.
    """
    if colors is None:
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_bars = len(data)
    bar_width = total_width / n_bars
    bars = []
    for i, (name, values) in enumerate(data.items()):
        x_offset = (i - n_bars / 2) * bar_width + bar_width / 2
        for x, y in enumerate(values):
            bar = ax.bar(x + x_offset, y, width=bar_width * single_width, color=colors[i % len(colors)])
        bars.append(bar[0])
    if legend:
        ax.legend(bars, data.keys())
    if ticks is not None:
        ticks.insert(0, None)
        ax.set_xticklabels(ticks)
    plt.ylabel('Average epoch duration in (sec)')
    plt.xlabel('Batch size')
    plt.title(title)



def save_times(time_list, file_name):
    with open(file_name, 'w') as file:
        file.write(",".join(time_list))

def dev_name(x):
    if x == "cpu":
        return "CPU"
    return "GPU"

def get_average(file_name):
    with open(file_name, 'r') as file:
        content = file.read()
        content = list(reversed(list(map(float, content.split(',')))))
        nofcontent = len(content)
        avg = 0
        first = content.pop(0)
        while content:
            second = content.pop(0)
            avg += (first-second)
            first = second
        return avg/nofcontent


if __name__ == "__main__":

    batch_ticks = [16, 32, 64, 128, 256, 512, 1024, 2048]
    # Usage example:
    data_time_test1 = {
        "(PC1) GPU": [get_average("results/PC1_test1_GPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        "(PC2) GPU": [get_average("results/PC2_test1_GPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        #"(PC1) CPU": [get_average("results/PC1_test1_CPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        #"(PC2) CPU": [get_average("results/PC2_test1_CPU_batch_{:}.txt".format(b)) for b in batch_ticks]
    }

    data_time_test2 = {
        "(PC1) GPU": [get_average("results/PC1_test2_GPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        "(PC2) GPU": [get_average("results/PC2_test2_GPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        "(PC1) CPU": [get_average("results/PC1_test2_CPU_batch_{:}.txt".format(b)) for b in batch_ticks],
        "(PC2) CPU": [get_average("results/PC2_test2_CPU_batch_{:}.txt".format(b)) for b in batch_ticks]
    }

    fig, ax = plt.subplots()
    bar_plot(ax, data_time_test2, total_width=.8, single_width=.9, ticks=batch_ticks, title="Test 2")
    plt.savefig("test2_all.png", dpi=128)
    #plt.show()
