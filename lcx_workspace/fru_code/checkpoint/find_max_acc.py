import time
import argparse

def find_max(file):
    f = open(file, 'r')
    line = f.readline()
    max_1 = 0
    epoch = 0
    while line != '':
        words = line.split(' ')
        top1 = float(words[2][:-1])
        top5 = float(words[4][:-1])
        epoch = epoch + 1
        if top1 > max_1:
            max_1 = top1
        line = f.readline()
    return max_1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='set parama')
    parser.add_argument('--path', default='', type=str, help="path of log.txt")
    args = parser.parse_args()

    file = './' + args.path + '/log.txt'
    f = open(file, 'r')
    line = f.readline()
    max_1 = 0
    max_5 = 0
    epoch = 0
    max_epoch = 0
    while line != '':
        words = line.split(' ')
        top1 = float(words[2][:-1])
        top5 = float(words[4][:-1])
        epoch = epoch + 1
        if top1 > max_1:
            max_1 = top1
            max_5 = top5
            max_epoch = epoch
        line = f.readline()
    print(max_1, max_5, max_epoch)
