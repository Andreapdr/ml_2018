import csv
import numpy as np


def from_text_to_csv(text_path, csv_path):
    with open(text_path, 'r') as in_file:
        data = in_file.read()
        l = data.split("\n")
        out = list()
        for elem in l:
            out.append(elem[1:14].split(" "))
        with open(csv_path, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerows(out)


def convert_to_one_hot(csv_path, path_new):
    data = np.genfromtxt(csv_path, delimiter=",")
    dataList = list(data)
    final = list()
    for data_sample in dataList:
        dd = [[0], [0, 0, 0], [0, 0, 0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0]]
        for i in range(len(data_sample)):
            temp_index = int(data_sample[i]) - 1
            if i == 0:
                value = data_sample[0]
            else:
                value = 1
            dd[i][temp_index] = value
            if i == len(data_sample) - 1:
                temp = list()
                for encoded_seq in dd:
                    for encoded_bit in encoded_seq:
                        temp.append(encoded_bit)
                final.append(temp)
    write_one_hot(final, path_new)
    return final


def write_one_hot(data, path_new):
    with open(path_new, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)


def get_dataset(name_csv):
    dataset_np = np.genfromtxt(name_csv, delimiter=",")
    return dataset_np


# from_text_to_csv("dataset/monk3/monk3test.txt", "dataset/monk3//monk3test.csv")
# convert_to_one_hot("dataset/monk3/monk3test.csv", "dataset/monk3/monk3test_onehot.csv")

# test merging
