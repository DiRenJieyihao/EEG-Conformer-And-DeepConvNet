import torch
import numpy as np
from torch.utils import data
from DeepConvNet import Deep4Net
import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from test import test
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(num_sub):
    X_train = r'./data/BCI/X{}_train.npy'.format(num_sub)
    Y_train = r'./data/BCI/Y{}_train.npy'.format(num_sub)
    X_test = r'./data/BCI/X{}_test.npy'.format(num_sub)
    Y_test = r'./data/BCI/Y{}_test.npy'.format(num_sub)
    print("I am loading {}.".format(X_train))
    print("I am loading {}.".format(Y_train))
    print("I am loading {}.".format(X_test))
    print("I am loading {}.".format(Y_test))
    X_train = np.load(X_train)
    Y_train = np.load(Y_train)
    X_test = np.load(X_test)
    Y_test = np.load(Y_test)

    X_train = X_train.reshape([288, 1, 22, -1])
    X_test = X_test.reshape([288, 1, 22, -1])
    Y_train = Y_train.reshape([-1, 1])
    Y_test = Y_test.reshape([-1, 1])

    print("X_train.shape:{}".format(X_train.shape))
    print("Y_train.shape:{}".format(Y_train.shape))
    print("X_test.shape:{}".format(X_test.shape))
    print("Y_test.shape:{}".format(Y_test.shape))

    return X_train, Y_train, X_test, Y_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DeepConvNet')
    parser.add_argument('--subject_id', type=int)
    args = parser.parse_args()

    print("subject_id:{}".format(args.subject_id))

    seed = 20200220
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True
    # model = Deep4Net(in_chans=22, n_classes=4, input_window_samples=1000, final_conv_length='auto').to(device)
    model = Deep4Net(in_chans=22, n_classes=4, input_window_samples=1000).to(device)
    model_root = 'models'

    optimizer = optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=5e-4,
                            amsgrad=False)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=50 - 1)

    X_train, Y_train, X_test, Y_test = get_data(args.subject_id)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(Y_train).long()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(Y_test).long()

    train_dataset = data.TensorDataset(X_train, y_train)
    test_dataset = data.TensorDataset(X_test, y_test)

    train_loader = data.DataLoader(
        dataset=train_dataset,  # torch TensorDataset format
        batch_size=16,  # mini batch size
        shuffle=True,
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,  # torch TensorDataset format
        batch_size=16,  # mini batch size
        shuffle=True,
    )

    print("\nthe length of train_loader:{}".format(len(train_loader)))
    print("the length of test_loader:{}".format(len(test_loader)))

    best_accu_t = 0.0
    accu_t_list = []
    for epoch in range(50):

        print(lr_scheduler.get_lr())

        train_loader_iter = iter(train_loader)
        model.train()

        for i in range(len(train_loader)):
            data_train = train_loader_iter.next()
            s_img, s_label = data_train

            s_img = s_img.to(device)
            s_label = s_label.to(device)
            s_label = torch.squeeze(s_label, 1)

            class_output = model(s_img)

            # loss_class = torch.nn.CrossEntropyLoss()
            # err_s_label = loss_class(class_output, s_label)

            # err_s_label = F.cross_entropy(class_output, s_label)
            loss_class = torch.nn.NLLLoss()
            err_s_label = loss_class(class_output, s_label)
            err = err_s_label

            model.zero_grad()
            err.backward()
            optimizer.step()
            lr_scheduler.step()

            if (i+1) % 18 == 0:
                print('\r epoch: %d, [iter: %d / all %d], err: %f' \
                                         % (epoch, i + 1, len(train_loader), err.data.cpu().numpy()))
            # sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err: %f' \
            #                  % (epoch, i + 1, len(train_loader), err.data.cpu().numpy()))
            # sys.stdout.flush()
            torch.save(model, '{0}/bcicompetition_model_epoch_current.pth'.format(model_root))

        # torch.random.set_rng_state(rng_state)

        # if (epoch+1) % 5 == 0:
        #     lr_scheduler.step()

        accu_t = test(test_loader, epoch)
        # accu_t_list.append(accu_t)

        # accu_t = mean(accu_t_list)
        print('\nTest Accuracy of the %s dataset: %f\n' % ('bcicompetition', accu_t))
        if accu_t > best_accu_t:
            best_accu_t = accu_t
        torch.save(model, '{0}/bcicompetition_model_epoch_best.pth'.format(model_root))
    print("Subject {}'s best accuracy is {}.".format(args.subject_id, best_accu_t))
