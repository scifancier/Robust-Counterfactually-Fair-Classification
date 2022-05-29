import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import numpy as np


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


criterion = nn.CrossEntropyLoss()


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def full_train(epoch, model, train_loader, optimizer, scheduler):
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    
    # print('\n==== Epoch:{0} ===='.format(epoch))

    
    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for _, (az, z, w, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        x = torch.cat((az, w), dim=1)
        out = model(x)
        loss = F.cross_entropy(out, by, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        
        prediction = torch.argmax(out, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    loss = loss.detach().item()
    acc = (correct / total).detach().item()
    
    # print('[Train] Loss: ', loss)
    # print('[Train] ACC: ', acc)

    return loss, acc


def full_evaluate(eval_loader, model, type):
    if torch.cuda.is_available():
        model = model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    model.eval()

    for _, (az, z, w, by) in enumerate(eval_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        x = torch.cat((az, w), dim=1)
        # x = torch.unsqueeze(x, 1)
        out = model(x)

        prediction = torch.argmax(out, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    acc = (correct / total).detach().item()
    # if type == 'val':
    #     print('\n---- Validation ----')
    #     print('[Validation] ACC: ', acc)
    # if type == 'test':
    #     print('\n---- Evaluation ----')
    #     print('[Test] ACC: ', acc)

    return acc


def vaz_train(epoch, model, train_loader, optimizer, scheduler):
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    for _, (az, z, w, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        x = z
        out = model(x)
        loss = F.cross_entropy(out, by, reduction="mean")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        prediction = torch.argmax(out, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    loss = loss.detach().item()
    acc = (correct / total).detach().item()

    # print('[Train] Loss: ', loss)
    # print('[Train] ACC: ', acc)


def vaz_evaluate(eval_loader, model, type):
    if torch.cuda.is_available():
        model = model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    model.eval()

    for _, (az, z, w, by) in enumerate(eval_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        x = z
        out = model(x)

        prediction = torch.argmax(out, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    acc = (correct / total).detach().item()
    # if type == 'val':
    #     print('\n---- Validation ----')
    #     print('[Validation] ACC: ', acc)
    # if type == 'test':
    #     print('\n---- Evaluation ----')
    #     print('[Test] ACC: ', acc)

    return acc

def mpz_train(epoch, model, train_loader, optimizer, scheduler):
    if torch.cuda.is_available():
        model = model.cuda()
    model.train()

    for _, (az, z, w, by) in enumerate(train_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        inputs, targets_a, targets_b, lam = mixup_data(z, by)

        inputs, targets_a, targets_b = map(Variable, (inputs,
                                                      targets_a, targets_b))

        out = model(inputs)
        loss = mixup_criterion(criterion, out, targets_a, targets_b, lam)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)


def mpz_evaluate(eval_loader, model, type):
    if torch.cuda.is_available():
        model = model.cuda()

    correct = torch.zeros(1).squeeze().cuda()
    total = torch.zeros(1).squeeze().cuda()

    model.eval()

    for _, (az, z, w, by) in enumerate(eval_loader):
        if torch.cuda.is_available():
            with torch.no_grad():
                az = az.cuda().float()
                z = z.cuda().float()
                w = w.cuda().float()
                by = by.cuda()

        x = z
        out = model(x)

        prediction = torch.argmax(out, 1)
        correct += (prediction == by).sum().float()
        total += len(by)

    acc = (correct / total).detach().item()
    # if type == 'val':
    #     print('\n---- Validation ----')
    #     print('[Validation] ACC: ', acc)
    # if type == 'test':
    #     print('\n---- Evaluation ----')
    #     print('[Test] ACC: ', acc)

    return acc
