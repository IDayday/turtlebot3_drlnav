import numpy as np

# del_num = real_scan.shape[0] - 360
# gap = real_scan.shape[0]//del_num

# del_index = real_scan[np.arange(0,real_scan.shape[0],gap+1)]
# print("left num", real_scan.shape[0] - del_index.shape[0])
# del_1 = np.delete(real_scan, del_index)
# del_num_2 = del_1.shape[0] - 360
# del_index_2 = np.arange(0, del_1.shape[0])[np.random.randint(0, del_1.shape[0], del_num_2)]
# outcomt = np.delete(del_1, del_index_2)
# print(outcomt.shape[0])


def filter_scan(real_scan):
    length = real_scan.shape[0]
    while length != 360:
        del_num = length - 360
        gap = length//del_num + 1
        del_index = np.arange(0, length, gap)
        outcome = np.delete(real_scan, del_index)
        length = outcome.shape[0]
        real_scan = outcome
    return outcome

for i in range(100):
    real_scan = np.arange(0,np.random.randint(370,580))
    outcome = filter_scan(real_scan)
    if outcome.shape[0] != 360:
        print("error", real_scan.shape[0])