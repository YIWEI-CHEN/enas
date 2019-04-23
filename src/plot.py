import matplotlib.pyplot as plt
import numpy as np

color_seq = [
    '#1f77b4', '#ff7f0e', '#2ca02c',
    '#ff9896', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
    '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]
if __name__ == '__main__':
    files = [
        # '/home/yiwei/enas/0423/acc_micro_final_clean_init_arch',
        '/home/yiwei/enas/0423/acc_micro_final_clean_final_arch',
        # '/home/yiwei/enas/0423/acc_micro_final_uniform_noise_08_init_arch',
        '/home/yiwei/enas/0423/acc_micro_final_uniform_noise_08_final_arch',
        # '/home/yiwei/enas/0423/acc_micro_final_uniform_noise_08_robust_init_arch_v2',
        '/home/yiwei/enas/0423/acc_micro_final_uniform_noise_08_robust_final_arch',
    ]
    labels = [
        # 'Clean Search + Init Arch.',
        'Clean Search + Final Arch.',
        # 'Noise Search + Init Arch. (0.28)',
        'Noise Search + Final Arch.',
        # 'Noise + Robust + Search + Init Arch. (0.28)',
        'Noise Search + Final Arch + Robust Loss (alpha=1.0)',
    ]
    for idx, file_name in enumerate(files):
        with open(file_name) as f:
            content = f.readlines()
        line_num = len(content)
        y_axis = map(float, content)
        x_axis = np.arange(1, line_num + 1)
        plt.plot(x_axis, y_axis, color=color_seq[idx], label=labels[idx])
    plt.legend(labels)
    plt.title('0423 Uniform Noise (Keep = 0.28)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('enas_micro_final_train_uniform_08')
    # plt.show()


