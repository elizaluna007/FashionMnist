import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import precision_score, recall_score


def train(args, epoch, dense_net, loader_train, optimizer, pfile_train):
    dense_net.train()
    nProcessed = 0
    nTrain = len(loader_train.dataset)
    i = 0
    for batch_idx, (data, target) in enumerate(loader_train):
        i = i+1
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = dense_net(data)
        loss = F.nll_loss(output, target)
        # make_graph.save('/tmp/t.dot', loss.creator); assert(False)
        loss.backward()
        optimizer.step()
        nProcessed += len(data)
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        incorrect = pred.ne(target.data).cpu().sum()
        err = 100.*incorrect/len(data)
        partialEpoch = epoch + batch_idx / len(loader_train) - 1
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tError: {:.6f}'.format(
            partialEpoch, nProcessed, nTrain, 100. *
            batch_idx / len(loader_train),
            loss.item(), err))

        pfile_train.write('{},{},{}\n'.format(partialEpoch, loss.item(), err))
        pfile_train.flush()

def test(args, epoch, dense_net, loader_test, pfile_test):
    dense_net.eval()
    test_loss = 0
    incorrect = 0
    all_preds = []
    all_targets = []

    for data, target in loader_test:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = dense_net(data)
        test_loss += F.nll_loss(output, target).item()
        pred = output.argmax(dim=1)
        incorrect += pred.ne(target).sum().item()

        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

    test_loss /= len(loader_test)
    accuracy = 1.0 - (incorrect / len(loader_test.dataset))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(
        test_loss, accuracy * 100))

    pfile_test.write('{},{},{},'.format(epoch, test_loss, accuracy))

    precision = precision_score(all_targets, all_preds, average=None)
    recall = recall_score(all_targets, all_preds, average=None)

    for class_idx in range(len(precision)):
        print('Class {}: Precision: {:.2f}%, Recall: {:.2f}%'.format(
            class_idx, precision[class_idx] * 100, recall[class_idx] * 100))
    pfile_test.write('precision,{},recall,{}\n'.format(
        ','.join(map(str, precision)), ','.join(map(str, recall))))
    pfile_test.flush()


def adjust_optimizer(alg, alg_param, epoch):
    if alg == 'sgd':
        if epoch < 150:
            lr = 1e-1
        elif epoch == 150:
            lr = 1e-2
        elif epoch == 225:
            lr = 1e-3
        else:
            return

        for param_group in alg_param.param_groups:
            param_group['lr'] = lr
