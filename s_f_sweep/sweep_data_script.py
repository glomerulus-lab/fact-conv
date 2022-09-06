f_list = [0.1, 2.0]
s_list = [1, 2, 3]
for f in f_list:
    for s in s_list:
        new_name = "s_"+str(s)+"_f_"+str(f)
        print("python3 sweep_data.py --s={} --f={} ".format(s, f))