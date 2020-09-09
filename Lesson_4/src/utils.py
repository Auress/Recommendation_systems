import pandas as pd
import numpy as np


def prefilter_items(data, take_n_popular=5000):
    """Предфильтрация товаров"""

    # 0. Удаляем позиции у которых количество = 0
    idx = data.loc[data['quantity'] == 0].index
    data.drop(idx, inplace = True)
    data.reset_index(drop=True, inplace=True)
    data['usd_per_unit'] = data['sales_value'] / data['quantity']

    # 1.2. Удаление товаров, со средней ценой < 1$ and > 30$
    df_grouped = data.groupby(by=['item_id']).mean()
    idx_less_1 = data.loc[data['item_id'].isin(df_grouped.loc[df_grouped['usd_per_unit'] < 1].index)].index
    idx_more_30 = data.loc[data['item_id'].isin(df_grouped.loc[df_grouped['usd_per_unit'] > 30].index)].index
    data = data.loc[~data.index.isin(np.concatenate([idx_less_1, idx_more_30]))]
    data.reset_index(drop=True, inplace=True)

    # 3. Придумайте свой фильтр (убраны товары, которые не продавались 1 год)
    df_grouped = data.groupby(by=['item_id']).max().sort_values('day')
    idx_sold_12m = data.loc[data['item_id'].isin(df_grouped.loc[df_grouped['day'] > (df_grouped['day'].max() - 366)].index)].index
    data = data.loc[data.index.isin(idx_sold_12m)]
    data.reset_index(drop=True, inplace=True)

    # 4. Выбор топ-N самых популярных товаров (N = take_n_popular)
    df_grouped = data.groupby(by=['item_id']).sum().sort_values('quantity', ascending = False)
    idx_top_k = data.loc[data['item_id'].isin(df_grouped.head(take_n_popular).index)].index
    data = data.loc[data.index.isin(idx_top_k)]
    data.reset_index(drop=True, inplace=True)

    return data
