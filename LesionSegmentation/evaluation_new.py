import numpy as np
from collections import Counter

def get_ious(labels, outputs):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab > 0, 1, 0)
        out = np.where(out > 0, 1, 0)

        lab_sum = np.sum(lab)
        out_sum = np.sum(out)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue

        intersection = np.sum(lab * out)
        union = lab_sum + out_sum - intersection
        iou = intersection / union  

        metric += iou

    metric /= batch_size
    return round(metric, 4)

def get_dices(labels, outputs):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab > 0, 1, 0)
        out = np.where(out > 0, 1, 0)

        lab_sum = np.sum(lab)
        out_sum = np.sum(out)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue


        intersection = np.sum(lab * out)
        union = lab_sum + out_sum
        dice = intersection*2 / union

        metric += dice

    metric /= batch_size

    return round(metric, 4)

def get_sens(labels, outputs):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab > 0, 1, 0)
        out = np.where(out > 0, 1, 0)

        lab_sum = np.sum(lab)
        out_sum = np.sum(out)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue


        tp = np.sum(lab * out)

        sen = tp / lab_sum

        metric += sen

    metric /= batch_size

    return round(metric, 4)


def get_spes(labels, outputs):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab > 0, 1, 0)
        out = np.where(out > 0, 1, 0)

        ones = np.ones_like(lab)
        lab_b = ones - lab
        out_b = ones - out

        lab_sum = np.sum(lab_b)
        out_sum = np.sum(out_b)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue

        tp = np.sum(lab_b * out_b)

        lab_sum = np.sum(lab_b)

        spe = tp / lab_sum

        metric += spe

    metric /= batch_size

    return round(metric, 4)


def cal_confu_matrix(label, predict, class_num):
    confu_list = []
    for i in range(class_num):
        c = Counter(predict[np.where(label == i)])
        single_row = []
        for j in range(class_num):
            single_row.append(c[j])
        confu_list.append(single_row)
    return np.array(confu_list).astype(np.int32)


def get_mious(cf_mtx):
    mIou = np.diag(cf_mtx) / (np.sum(cf_mtx, axis=1) + np.sum(cf_mtx, axis=0) - np.diag(cf_mtx))
    mIou = mIou[1:]
    mIou = np.nanmean(mIou)
    return mIou

def get_mious_pro(cf_mtx):
    intersection = np.diag(cf_mtx) 
    union = np.sum(cf_mtx, axis=1) + np.sum(cf_mtx, axis=0) - np.diag(cf_mtx) 
    IoU = intersection / union  
    mIoU = np.nanmean(IoU) 
    return mIoU

def get_pa(cf_mtx):
    acc = np.diag(cf_mtx).sum() / cf_mtx.sum()
    return acc

def get_mPa(cf_mtx):
    # classAcc = get_pa(cf_mtx)
    # meanAcc = np.nanmean(classAcc)
    classAcc = np.diag(cf_mtx) / cf_mtx.sum(axis=1)
    classAcc = classAcc[1:]
    mPa = np.nanmean(classAcc)
    return mPa

def get_cpa(cf_mtx):
    classAcc = np.diag(cf_mtx) / cf_mtx.sum(axis=1)
    return classAcc 

def get_dices_pro(labels, outputs, index=1):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab == index, 1, 0)
        out = np.where(out == index, 1, 0)

        lab_sum = np.sum(lab)
        out_sum = np.sum(out)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue

        intersection = np.sum(lab * out)
        union = lab_sum + out_sum
        dice = intersection*2 / union

        metric += dice

    metric /= batch_size

    return round(metric, 4)


def get_dices_pros(labels, outputs):
    batch_size = labels.shape[0]
    metric = 0.0
    for batch in range(batch_size):
        lab, out = labels[batch], outputs[batch]

        lab = lab.detach().cpu().squeeze().numpy()
        out = out.detach().cpu().squeeze().numpy()

        lab = np.where(lab > 0, 1, 0)
        out = np.where(out > 0, 1, 0)

        lab_sum = np.sum(lab)
        out_sum = np.sum(out)

        if lab_sum == 0:
            metric += (out_sum == 0)
            continue

        intersection = np.sum(lab * out)
        union = lab_sum + out_sum
        dice = intersection*2 / union

        metric += dice

    metric /= batch_size

    return round(metric, 4)

if __name__ == '__main__':
    output = [[1,1,1,1,2,2,3,3], [1,1,1,1,2,2,3,3]]
    label = [[0,0,1,1,2,2,3,3], [0,0,1,1,2,2,3,3]]

    # print(get_ious(np.array(label), np.array(output)))
    # print(get_dices(np.array(label), np.array(output)))
    # print(get_sens(np.array(label), np.array(output)))
    # print(get_spes(np.array(label), np.array(output)))
    m = cal_confu_matrix(np.array(label), np.array(output), 4)
    # print('m -->', m)
    # print('miou -->', get_mious(m))
    # print('miou pro -->', get_mious_pro(m))
    # print('pa -->', get_pa(m))
    print('cpa -->', get_cpa(m))
    # print('mpa -->', get_mPa(m))




