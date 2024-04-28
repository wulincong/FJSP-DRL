import matplotlib.pyplot as plt

def plot_funetuning(data, title, y_label="Makespan", label=None):
    # Labels for the lines
    labels = ["DAN_baseline", "MAML", "pre_training", "random"]
    x = range(1, len(data[0]) + 1)
    # Plotting the lines
    for i, row in enumerate(data):
        if label: plt.plot(x, row, label=labels[i], linewidth=0.5)
        else: plt.plot(x, row, linewidth=0.5)
        plt.scatter(x, row, s=10)  # s 参数控制点的大小
    plt.xlabel("Finetuning Epoch") 
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()