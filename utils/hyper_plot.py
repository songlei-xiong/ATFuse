from cProfile import label
import matplotlib.pyplot as plt
# impor

named_data = {
    'P': {'label': [1,	4,	8,	16,	18],
          'metrics':
          {'ERGAS': [1.342,	1.233,	1.231,	1.205,	1.243],
           'SAM':    [2.148,	2.053,	2.095,	2.049,	2.11],
           'Q2n':     [0.963,	0.962,	0.962,	0.965,	0.962],
           '#Params':   [3.252,	3.295,	3.917,	13.779,	20.1]}
          },
    "H": {'label': [1, 2, 4, 6, 8, 120],
          "metrics": {
        "ERGAS": [1.264, 1.341, 1.207, 1.306, 1.233, 7.614],
        "SAM": [2.197, 2.273, 2.102, 2.092, 2.053, 4.022],
        "Q2n": [0.962, 0.958, 0.963, 0.962, 0.962, 0.757],
        '#Params': [3.257, 3.263, 3.274, 3.285, 3.295, 3.905]}
    },
    "K": {'label': [3, 5, 7, 9, 11],
          "metrics": {
        "ERGAS": [1.253, 1.233, 1.352, 1.233, 1.254],
        "SAM": [2.069, 2.105, 2.187, 2.053, 2.142],
        "Q2n": [0.961, 0.961, 0.962, 0.962, 0.957],
        "#Params": [3.209, 3.228, 3.257, 3.295, 3.343],
    }
    },

    "D": {'label': [30, 60, 90, 120, 180],
                'metrics': {
        "ERGAS": [3.379, 1.566, 1.303, 1.233, 1.325],
        "SAM": [3.527, 2.385, 2.12, 2.053, 2.138],
        'Q2n': [0.918, 0.957, 0.963, 0.962, 0.962],
        '#Params': [0.292, 0.917, 1.918, 3.295, 7.179]
    }
    }
}

for metric in ('ERGAS', 'SAM', 'Q2n'):
    for hyper_para, value in named_data.items():
        plt.plot(value[metric], label=hyper_para)
        plt.text(range(1, len(value[metric])))