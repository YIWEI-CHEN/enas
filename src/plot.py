import matplotlib.pyplot as plt
import numpy as np

color_seq = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                  '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

if __name__ == '__main__':
    files = {
        # 'clean_init': '/home/yiwei/mac_book/enas/acc_micro_final_clean_init_arch',
        # 'clean_final': '/home/yiwei/mac_book/enas/acc_micro_final_clean_final_arch',
        # 'noise_init': '/home/yiwei/mac_book/enas/acc_micro_final_uniform_noise_08_init_arch',
        # 'noise_final': '/home/yiwei/mac_book/enas/acc_micro_final_uniform_noise_08_final_arch',
        # 'robust_init': 'Noise Search + Robust + Init Arch',
        # 'robust_final': 'Noise Search + Robust + Final Arch',
        # 'clean_final': '/home/yiwei/mac_book/enas/enas_qq/macro_final_fit/acc_enas_macro_clean_final_fit',
        # 'noise_init': '/home/yiwei/mac_book/enas/enas_qq/macro_final_fit/acc_enas_macro_permute_init_fit',
        # 'noise_final': '/home/yiwei/mac_book/enas/enas_qq/macro_final_fit/acc_enas_macro_permute_final_fit',
        'clean_final': '/home/yiwei/mac_book/enas/auto_keras/acc_clean_7539',
        # 'noise_final': '/home/yiwei/mac_book/enas/auto_keras/acc_perm_7114',
        # 'robust_final': '/home/yiwei/mac_book/enas/auto_keras/acc_perm_robust_7321',
        'noise_final': '/home/yiwei/mac_book/enas/auto_keras/acc_sym_055_7329',
        'robust_final': '/home/yiwei/mac_book/enas/auto_keras/acc_sym_055_robust_7215',

    }
    labels = {
        'clean_init': 'Clean Search + Init Arch',
        'clean_final': 'Clean Search + Final Arch',
        'noise_init': 'Noise Search + Init Arch',
        'noise_final': 'Noise Search + Final Arch',
        'robust_init': 'Noise Search + Robust + Init Arch',
        'robust_final': 'Noise Search + Robust + Final Arch',
    }
    colors = {
        'clean_init': color_seq[0],
        'clean_final': color_seq[2],
        'noise_init': color_seq[6],
        'noise_final': color_seq[4],
        'robust_init': color_seq[8],
        'robust_final': color_seq[10],

    }

    legend = []
    for key, file_name in files.iteritems():
        with open(file_name) as f:
            content = f.readlines()
        line_num = len(content)
        y_axis = map(float, content)
        x_axis = np.arange(1, line_num + 1)
        if len(y_axis) > 300:
            y_axis = y_axis[:300]
            x_axis = x_axis[:300]
        plt.plot(x_axis, y_axis, color=colors[key], label=labels[key])
        legend.append(labels[key])

    plt.legend(legend)
    plt.title('Uniform Noise 0.5 (Auto-Keras)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('auto_keras_sym_055_uni_05')
    # plt.show()

