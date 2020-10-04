import pandas as pd
import numpy as np


def hit_rate(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def hit_rate_at_k(recommended_list, bought_list, k=5):

    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])

    flags = np.isin(bought_list, recommended_list)

    hit_rate = (flags.sum() > 0) * 1

    return hit_rate


def precision(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if ~np.isnan(recommended_list.sum()):
        flags = np.isin(bought_list, recommended_list)
        precision_result = flags.sum() / len(recommended_list)
    else:
        precision_result = 0

    return precision_result


def precision_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    bought_list = bought_list  # Тут нет [:k] !!
    recommended_list = recommended_list[:k]

    flags = np.isin(bought_list, recommended_list)

    precision_result = flags.sum() / len(recommended_list)

    return precision_result


def money_precision_at_k(recommended_list, bought_list, prices, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])

    prices_recommended = np.array([prices[i] for i in recommended_list])
    prices_bought = np.array([prices[i] if i in prices.keys() else 0 for i in bought_list])

    flags = np.isin(bought_list, recommended_list)
    precision = np.dot(flags, prices_bought) / np.dot(np.ones(k), prices_recommended)

    return precision


def recall(recommended_list, bought_list):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    if ~np.isnan(bought_list.sum()):
        flags = np.isin(bought_list, recommended_list)
        recall = flags.sum() / np.shape(bought_list)[0]
    else:
        recall = 0

    return recall


def recall_at_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])

    flags = np.isin(bought_list, recommended_list)

    recall = flags.sum() / len(bought_list)

    return recall


def money_recall_at_k(recommended_list, bought_list, prices_recommended, prices_bought, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list[:k])
    prices_recommended_list = prices_recommended[:k]
    prices_bought_list = prices_bought

    flags = np.isin(recommended_list, bought_list)

    recall = sum(flags * prices_recommended_list) / sum(prices_bought_list)

    return recall


def ap_k(recommended_list, bought_list, k=5):
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)

    flags = np.isin(recommended_list, bought_list)

    if sum(flags) == 0:
        return 0

    sum_ = 0
    for i in range(1, k + 1):

        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i)
            sum_ += p_k

    result = sum_ / sum(flags)

    return result


def reciprocal_rank(recommended_item, bought_list):
    bought_list = np.array(bought_list)

    try:
        rank = 1 / (np.where(bought_list == recommended_item)[0][0] + 1)
    except IndexError:
        rank = 0

    return rank


def mean_reciprocal_rank(recommended_list, bought_list, k):
    recommended_list = np.array(recommended_list[:k])
    result = 0

    for item in recommended_list:
        result += reciprocal_rank(recommended_item=item, bought_list=bought_list)

    result /= k

    return result
