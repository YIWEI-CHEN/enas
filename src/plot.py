import matplotlib.pyplot as plt
import numpy as np

color_seq = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
    '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
    '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]
if __name__ == '__main__':
    files = [
        'num_filter_std_micro_final_clean_final_archs',
        'num_filter_std_micro_final_noise_final_arch',
        'num_filter_std_micro_final_noise_init_arch'
    ]
    labels = [
        'Clean Search Final Arch.',
        'Noise Search Final Arch.',
        'Noise Search Init Arch.',
    ]
    for idx, file_name in enumerate(files):
        with open(file_name) as f:
            content = f.readlines()
        line_num = len(content)
        y_axis = map(float, content)
        x_axis = np.arange(1, line_num + 1)
        plt.plot(x_axis, y_axis, color=color_seq[idx], label=labels[idx])
    plt.legend(labels)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('enas_micro_final_train')
    plt.show()


