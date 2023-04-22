import os
import torch.utils.data

device = "cuda" if torch.cuda.is_available() else "cpu"


def test(dataloader, epoch):
    model_root = 'models'

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'bcicompetition_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    my_net = my_net.to(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0
    value = 0.

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        t_img = t_img.to(device)
        t_label = t_label.to(device)

        class_output = my_net(t_img)

        loss_class = torch.nn.NLLLoss()
        err_s_label = loss_class(class_output, torch.squeeze(t_label, 1))
        err = err_s_label

        # sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err: %f' \
        #                  % (epoch, i + 1, len_dataloader, err.data.cpu().numpy()))
        # sys.stdout.flush()
        # sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err: %f' \
        #                          % (epoch, i + 1, len_dataloader, err.data.cpu().numpy()))
        if (i+1) % 18 == 0:
            print('\r epoch: %d, [iter: %d / all %d], err: %f' \
             % (epoch, i + 1, len_dataloader, err.data.cpu().numpy()))
        # sys.stdout.flush()

        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        # value += accuracy_score(t_label.data.view_as(pred), pred)
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total
    # print("value:{}".format(value/i))

    return accu
