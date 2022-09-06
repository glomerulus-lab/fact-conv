import numpy as np

f_arr = np.array([0.1, 0.5, 1, 2, 3, 5, 7])
s_arr = np.array([1, 2, 3, 4, 5, 6, 10])

num_combinations = len(f_arr) * len(s_arr)
i = 0
for f in f_arr:
    for s in s_arr:
        i += 1
        if i <= num_combinations / 2:
            for trial in range(3):
                print("Trial: {} s: {} f: {}".format(trial+1, s, f))
                print("python3 run_model.py --device=0 --s={} --f={} --trial={} --name='s_{}_f_{}'".format(s, f, trial+1, s, f))
        else:
            for trial in range(3):
                print("Trial: {} s: {} f: {}".format(trial+1, s, f))
                print("python3 run_model.py --device=1 --s={} --f={} --trial={} --name='s_{}_f_{}'".format(s, f, trial+1, s, f))