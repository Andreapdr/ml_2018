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


def simple_plot(task, network, lr, momentum):
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
    plt.plot(cord_x, cord_y, label=f"Error Rate Training")
    plt.plot(cord_x_test, cord_y_test, label="Error Rate Validation", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.grid(True)
    plt.legend()
    plt.show()

    if task == "monk":
        plt.title(f"Accuracy Plot \nlr: {lr}, momentum: {momentum}")
        cord_x_acc = list()
        cord_y_acc = list()
        cord_x_acc_valid = list()
        cord_y_acc_valid = list()
        for elem in network.accuracy_list:
            cord_x_acc.append(elem[0])
            cord_y_acc.append(elem[1])
        for elem in network.validation_accuracy_list:
            cord_x_acc_valid.append(elem[0])
            cord_y_acc_valid.append(elem[1])
        plt.plot(cord_x_acc, cord_y_acc, label="Accuracy on training")
        plt.plot(cord_x_acc_valid, cord_y_acc_valid, label="Accuracy on test", linestyle="--")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.legend()
        plt.show()



def simple_plot_test(network, lr, momentum):
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
    plt.plot(cord_x, cord_y, label=f"Error Rate Training")
    plt.plot(cord_x_test, cord_y_test, label="Error Rate Validation")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.title(f"Accuracy Plot \nlr: {lr}, momentum: {momentum}")
    cord_x_acc = list()
    cord_y_acc = list()
    cord_x_acc_valid = list()
    cord_y_acc_valid = list()
    for elem in network.accuracy_list:
        cord_x_acc.append(elem[0])
        cord_y_acc.append(elem[1])
    for elem in network.validation_accuracy_list:
        cord_x_acc_valid.append(elem[0])
        cord_y_acc_valid.append(elem[1])
    plt.plot(cord_x_acc, cord_y_acc, label="Accuracy on training")
    plt.plot(cord_x_acc_valid, cord_y_acc_valid, label="Accuracy on test")


def plot_multinetwork(network_list, lr, momentum, lambd, folds, architecture):
    avg_error_list_x = [0] * len(network_list[0].error_list)
    avg_error_list_y = [0] * len(network_list[0].error_list)
    avg_val_error_list_x = [0] * len(network_list[0].error_list)
    avg_val_error_list_y = [0] * len(network_list[0].error_list)

    for index, network in enumerate(network_list):
        plt.title(f"Error Function Plot \nlr: {lr}, momentum: {momentum}, lambda: {lambd}, {architecture}")
        cord_x = list()
        cord_y = list()
        cord_x_val = list()
        cord_y_val = list()
        for elem in network.error_list:
            cord_x.append(elem[0])
            cord_y.append(elem[1])
        for elem in network.validation_error_list:
            cord_x_val.append(elem[0])
            cord_y_val.append(elem[1])

        for i in range(len(cord_x)):
            avg_error_list_x[i] = cord_x[i]
            avg_error_list_y[i] += cord_y[i]
            avg_val_error_list_x[i] = cord_x_val[i]
            avg_val_error_list_y[i] += cord_y_val[i]

        # plt.plot(cord_x, cord_y, alpha=0.3, label=f"Error Rate Training {index+1}")
        # plt.plot(cord_x_val, cord_y_val, alpha=0.3, label="Error Rate Validation")

    for i in range(len(avg_error_list_y)):
        avg_error_list_y[i] = avg_error_list_y[i]/folds
        avg_val_error_list_y[i] = avg_val_error_list_y[i]/folds

    plt.plot(avg_error_list_x, avg_error_list_y, c="red", label="Average TR")
    plt.plot(avg_val_error_list_x, avg_val_error_list_y, c="blue", label="Average VAL", linestyle="--")

    plt.grid(True)
    plt.legend()
    temp = str(architecture)
    plt.savefig("/home/andrea/sviluppo/ml_2018/plots/" + temp + ".png")
    plt.show()


def k_fold(data, folding):
    if folding == 1:
        len_training = (len(data)//100) * 70
        training = data[:len_training]
        validation = data[len_training:]
        return [training], [validation]
    else:
        len_data = len(data)
        fold_len = len_data // folding
        folded = []
        train = []
        validation = []
        t = 0
        for i in range(fold_len, len_data, fold_len):
            if i == fold_len * folding:
                folded.append(data[t:])
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
