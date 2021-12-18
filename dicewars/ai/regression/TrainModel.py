import random

import pandas as pd
import torch
import torch.nn.functional as F

import Model as m


# torch.manual_seed(0)
# np.random.seed(0)
# random.seed(0)


# first,second,third,fourth,fifth,sixth,sevent,eighth,win

def shuffle_data(basic, shuffled):
    with open(basic, 'r') as r, open(shuffled, 'w') as w:
        data = r.readlines()
        header, rows = data[0], data[1:]
        random.shuffle(rows)
        rows = '\n'.join([row.strip() for row in rows])
        w.write(header + rows)


def threshold(x, t=0.65):
    if x < t:
        return 0
    else:
        return 1


def evaluate(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainset:
            X, y = data
            output = model(X)

            for idx, i in enumerate(output):
                if threshold(i) == y[idx]:
                    correct += 1
                total += 1
    return correct


def load_model():
    model = m.Model()
    model.load_state_dict(torch.load("xreset00_model"))
    model.eval()
    return model


def save_model(model):
    torch.save(model.state_dict(), "SUI_model")


if __name__ == '__main__':
    training_dataset = pd.read_csv("Datasets/shuffled_training_data.csv",
                                   names=["game_result", "enemies", "enemies_areas", "enemies_dice", "my_dice",
                                          "my_areas",
                                          "border_areas", "border_dice", "regions", "enemies_regions",
                                          "biggest_region"])
    test_dataset = pd.read_csv("Datasets/shuffled_test_data.csv",
                               names=["game_result", "enemies", "enemies_areas", "enemies_dice", "my_dice", "my_areas",
                                      "border_areas", "border_dice", "regions", "enemies_regions", "biggest_region"])
    training_data = training_dataset.copy()
    test_data = test_dataset.copy()

    training_win = training_data.pop("game_result")
    test_win = test_data.pop("game_result")
    # training_data.pop("enemies_dice")
    # training_data.pop("enemies_areas")
    # training_data.pop("my_areas")
    # training_data.pop("regions")
    # training_data.pop("border_dice")
    # test_data.pop("my_areas")
    # test_data.pop("enemies_areas")
    # test_data.pop("regions")
    # test_data.pop("border_dice")
    # test_data.pop("enemies_dice")

    train = list()
    test = list()
    for i, d in enumerate(training_data.values):
        train.append((torch.Tensor(d), training_win[i]))
    for i, d in enumerate(test_data.values):
        test.append((torch.Tensor(d), test_win[i]))

    trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
    testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

    net = m.Model()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    best_correct = 0
    for epocha in range(50):
        print(epocha)
        for i, data in enumerate(trainset):  # `data` is a batch of data
            X, y = data  # X is the batch of features, y is the batch of targets.
            net.zero_grad()  # sets gradients to 0 before loss calc. You will do this likely every step.
            output = net(X)  # pass in the reshaped batch (recall they are 28x28 atm)
            loss = F.binary_cross_entropy(output, y.float())  # calc and grab the loss value
            optimizer.zero_grad()
            loss.backward()  # apply this loss backwards thru the network's parameters
            optimizer.step()  #
        print(loss)
        new_correct = evaluate(net)
        if new_correct > best_correct:
            last_epocha = epocha
            best_correct = new_correct
            best_net = net
    print("last:", last_epocha)
    correct = 0
    total = 0

    with torch.no_grad():
        for data in trainset:
            X, y = data
            output = best_net(X)

            for idx, i in enumerate(output):
                if threshold(i) == y[idx]:
                    correct += 1
                total += 1
    print("Accuracy trainset: ", round(correct / total, 3))

    with torch.no_grad():
        for data in testset:
            X, y = data
            output = best_net(X)

            for idx, i in enumerate(output):
                if threshold(i) == y[idx]:
                    correct += 1
                total += 1

    print("Accuracy testset: ", round(correct / total, 3))

    print("Model's state_dict:")
    for param_tensor in best_net.state_dict():
        print(param_tensor, "\t", best_net.state_dict()[param_tensor])
    print(training_data.head())
    exit(0)
