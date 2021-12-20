import copy
import random

import pandas as pd
import torch
import torch.nn.functional as F

import os
import sys
import dicewars.ai.gf.Model as m


# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


# first,second,third,fourth,fifth,sixth,sevent,eighth,win

class TrainModel:
    def __init__(self):
        pass

    @staticmethod
    def shuffle_data(basic, shuffled):
        with open(basic, 'r') as r, open(shuffled, 'w') as w:
            data = r.readlines()
            header, rows = data[0], data[1:]
            random.shuffle(rows)
            rows = '\n'.join([row.strip() for row in rows])
            w.write(header + rows)

    @staticmethod
    def threshold(x, t=0.6):
        if x < t:
            return 0
        else:
            return 1

    def evaluate(self, model_f, set_f):
        correct_f = 0
        total_f = 0
        with torch.no_grad():
            for (X_f, y_f) in set_f:
                output_f = model_f(X_f)
                for idx, i_f in enumerate(output_f):
                    if self.threshold(i_f) == y_f[idx]:
                        correct_f += 1
                    total_f += 1
        if testset == set_f:
            print("Accuracy testset: ", round(correct_f / total_f, 3))
        else:
            print("Accuracy trainset: ", round(correct_f / total_f, 3))
        print("Correct: ", correct_f)
        print("Total: ", total_f)
        return round(correct_f / total_f, 3)

    @staticmethod
    def load_model():
        model = m.Model()
        # cwd = os.getcwd()
        # print(cwd)
        # for debug
        model.load_state_dict(torch.load("SUI_model"))
        # model.load_state_dict(torch.load("dicewars/ai/gf/SUI_model"))
        model.eval()
        return model

    @staticmethod
    def save_model(model):
        torch.save(model.state_dict(), "SUI_model")
        print("Saved new model to SUI_model file")

    @staticmethod
    def test_threshold():
        train_model = TrainModel()
        thresh_model = TrainModel.load_model()
        train_model.evaluate(thresh_model, trainset)
        train_model.evaluate(thresh_model, testset)
        exit(0)


if __name__ == '__main__':
    training_dataset = pd.read_csv("Datasets/shuffled_training_data.csv",
                                   names=["game_result", "enemies", "enemies_areas", "enemies_dice", "my_dice",
                                          "my_areas", "border_areas", "border_dice", "regions", "enemies_regions",
                                          "biggest_region"])
    test_dataset = pd.read_csv("Datasets/shuffled_test_data.csv",
                               names=["game_result", "enemies", "enemies_areas", "enemies_dice", "my_dice", "my_areas",
                                      "border_areas", "border_dice", "regions", "enemies_regions", "biggest_region"])
    training_data = training_dataset.copy()
    test_data = test_dataset.copy()

    training_win = training_data.pop("game_result")
    test_win = test_data.pop("game_result")

    train = list()
    test = list()
    for i, d in enumerate(training_data.values):
        train.append((torch.Tensor(d), training_win[i]))
    for i, d in enumerate(test_data.values):
        test.append((torch.Tensor(d), test_win[i]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    TrainModel.test_threshold()

    train_model = TrainModel()
    net = m.Model()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    best_correct = 0
    best_net = m.Model()
    loss = 0
    last_epocha = 0
    loss_array = []
    for epocha in range(50):
        print("\n", epocha)
        for i, data in enumerate(trainset):  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            optimizer.zero_grad()
            output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.binary_cross_entropy(output, y.float())  # calc and grab the loss value
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  #
        if epocha % 5 == 0:
            loss_array.append(loss)
        new_correct = train_model.evaluate(net, trainset)
        if new_correct > best_correct:
            last_epocha = epocha
            best_correct = new_correct
            best_net = copy.deepcopy(net)

    print("\nlast:", last_epocha)

    train_model.evaluate(best_net, trainset)
    train_model.evaluate(best_net, testset)

    # Pre ulozenie modelu do suboru SUI_model odkomentovat riadok pod
    train_model.save_model(best_net)

    print("\nBestModel's state_dict:")
    for param_tensor in best_net.state_dict():
        print(param_tensor, "\t", best_net.state_dict()[param_tensor])
    print("\n")

    for l in loss_array:
        print(l.item())

    exit(0)
