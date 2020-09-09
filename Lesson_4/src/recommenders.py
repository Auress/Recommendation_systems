import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """

    # data = pd.read_csv('../data/retail_train.csv')
    # item_features = pd.read_csv('../data/product.csv')

    def __init__(self, data, item_features, weighting=True):

        # Топ покупок каждого юзера
        self.top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
        self.top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.top_purchases = self.top_purchases[self.top_purchases['item_id'] != 999999]

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_values('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases[self.overall_top_purchases['item_id'] != 999999]
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid, self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        # Словарь {item_id: 0/1}. 0/1 - факт принадлежности товара к СТМ
        self.item_id_to_ctm = dict(zip(item_features['item_id'], item_features['brand'].isin(['Private']).astype(int)))

        # Own recommender обучается до взвешивания матрицы
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T

        self.model = self.fit(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data, val='quantity'):

        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values=val,  # Можно пробовать другие варианты
                                          aggfunc='count',
                                          fill_value=0
                                          )

        user_item_matrix = user_item_matrix.astype(float)  # необходимый тип матрицы для implicit

        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""

        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def get_similar_items_recommendation(self, user, filter_ctm=True, N=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        rec_model = self.model.similar_items(self.itemid_to_id[user], N=N)
        recs = [x[0] for x in rec_model][1:]  # получаем список рекомендаций
        rec_to_itemid = [self.id_to_itemid[x] for x in recs]  # переводим в изначальные id

        if filter_ctm:
            ctm_list = [self.item_id_to_ctm[x] for x in rec_to_itemid]  # Список является или нет товар СТМ

            try:
                idx = ctm_list.index(1)  # Берем первый товар СТМ
            except ValueError:
                idx = 0  # либо просто первый, если СТМ в списке не оказалось
        else:
            idx = 0

        res = rec_to_itemid[idx]

        return res

    def get_similar_users_recommendation(self, user, N_users=5, N_items_per_user=5, N=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""

        rec_model_users = self.model.similar_users(self.userid_to_id[user], N=N_users+1)
        recs_users = [x[0] for x in rec_model_users][1:]  # получаем список Юзеров

        total_recs = []
        for similar_user in recs_users:
            own = ItemItemRecommender(K=1, num_threads=4)  # K - кол-во билжайших соседей
            own.fit(csr_matrix(self.user_item_matrix).T.tocsr())

            recs = own.recommend(userid=self.userid_to_id[similar_user],  # Находим купленые ими товары
                                     user_items=csr_matrix(self.user_item_matrix).tocsr(),  # на вход user-item matrix
                                     N=N_items_per_user,
                                     filter_already_liked_items=False,
                                     filter_items=None,
                                     recalculate_user=False)
            total_recs.append(recs)

        total_recs = [item for sublist in total_recs for item in sublist]  # делаем общий список товаров
        total_recs = sorted(total_recs, key=lambda l: l[1], reverse=True)  # выбираем товары с большим мкором

        res = [self.id_to_itemid[x[0]] for x in total_recs][:N]  # берем нужное кол-во и переводим в изначальные id

        return res
