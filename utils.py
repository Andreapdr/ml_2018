import csv
import numpy as np
import matplotlib.pyplot as plt


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


def horror_plot(network_list, lr, momentum):
    for index, network in enumerate(network_list):
        plt.title(f"Error Function Plot \nlr: {lr}, momentum: {momentum}")
        cord_x = list()
        cord_y = list()
        cord_x_test = list()
        cord_y_test = list()
        for elem in network.error_list:
            cord_x.append(elem[0])
            cord_y.append(elem[1])
        for elem in network.validation_error_list:
            cord_x_test.append(elem[0])
            cord_y_test.append(elem[1])
        plt.plot(cord_x, cord_y, label=f"Error Rate Training {index+1}")
        # plt.plot(cord_x_test, cord_y_test, label="Error Rate Validation")
    plt.grid(True)
    plt.legend()
    plt.show()

    # plt.title(f"Accuracy")
    # acc_cord_x = list()
    # acc_cord_y = list()
    # acc_cord_x_test = list()
    # acc_cord_y_test = list()
    # for elem in network.accuracy_list:
    #     acc_cord_x.append(elem[0])
    #     acc_cord_y.append(elem[1])
    # for elem in network.validation_accuracy_list:
    #     acc_cord_x_test.append(elem[0])
    #     acc_cord_y_test.append(elem[1])
    # plt.plot(acc_cord_x, acc_cord_y, label="Accuracy Training")
    # plt.plot(acc_cord_x_test, acc_cord_y_test, label="Accuracy Validation")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


# TODO: what if data cannot be split evenly ?
def k_fold(data, folding):
    if folding == 1:
        len_training = len(data)//100 * 85
        training = data[:len_training]
        validation = data[len_training:]
        return [training], [validation]
    else:
        len_data = len(data)
        fold_len = len_data // folding
        fold_len2 = len_data // folding
        folded = []
        train = []
        validation = []
        t = 0
        for i in range(fold_len, len_data + fold_len, fold_len):
            if i == fold_len:
                folded.append(data[t:i+1])
                t = i
            else:
                folded.append(data[t:i])
                t = i
        for i in range(len(folded)):
            temp_folded = folded.copy()
            temp_folded.pop(i)
            temp_train = np.vstack(temp_folded)
            train.append(temp_train)
            validation.append(folded[i])
        return train, validation


# from_text_to_csv("dataset/monk3/monk3test.txt", "dataset/monk3//monk3test.csv")
# convert_to_one_hot("dataset/monk3/monk3test.csv", "dataset/monk3/monk3test_onehot.csv")
