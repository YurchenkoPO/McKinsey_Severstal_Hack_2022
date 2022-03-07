import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Model:
    def __init__(self, max_year, min_year):
        # To be continued
        self.finance_feat = [
            "Нематериальные активы",
            "Основные средства ",
            "Внеоборотные активы",
            "Дебиторская задолженность",
            "Оборотные активы",
            "Уставный капитал ",
            "Капитал и резервы",
            "Заёмные средства (долгосрочные)",
            "Долгосрочные обязательства",
            "Заёмные средства (краткосрочные)",
            "Кредиторская задолженность",
            "Краткосрочные обязательства",
            "Выручка",
            "Себестоимость продаж",
            "Прибыль (убыток) до налогообложения ",
            "Прибыль (убыток) от продажи",
        ]

        self.cat_cols = ["Наименование ДП"]

        self.max_year = max_year
        self.min_year = min_year

    def get_full_finance_feat_name(self, fin_feat, year):
        return str(year) + ', ' + fin_feat + ', RUB'

    def diff_finance_features(self, df, max_year, min_year):
        for fin_feat in self.finance_feat:
            for year in range(max_year - 1, min_year - 1, -1):
                df[fin_feat + f" ,разница за {max_year - year} год(а)"] = (
                    df[self.get_full_finance_feat_name(fin_feat, max_year)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                )
                scaler = MinMaxScaler()
                df[fin_feat + f" ,разница за {max_year - year} год(а)"] = scaler.fit_transform(
                    df[fin_feat + f" ,разница за {max_year - year} год(а)"].values.reshape(-1, 1)
                )

        return df

    def ratio_finance_features(self, df, max_year, min_year):
        for fin_feat in self.finance_feat:
            for year in range(max_year - 1, min_year - 1, -1):
                df[fin_feat + f" ,отношение за {max_year - year} год(а)"] = (
                    df[self.get_full_finance_feat_name(fin_feat, max_year)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                ) / df[self.get_full_finance_feat_name(fin_feat, max_year)]

                df[fin_feat + f" ,отношение за {max_year - year} год(а)"].fillna(0, inplace=True)

                df.replace([np.inf], 1, inplace=True)
                df.replace([-np.inf], -1, inplace=True)

                scaler = MinMaxScaler()
                df[fin_feat + f" ,отношение за {max_year - year} год(а)"] = scaler.fit_transform(
                    df[fin_feat + f" ,отношение за {max_year - year} год(а)"].values.reshape(-1, 1)
                )

        return df

    def cat_one_hot(self, df, cat_cols):
        df = pd.get_dummies(df, columns=cat_cols)

        return df

    def preprocessing_before_fitting(self, df):
        df = self.diff_finance_features(df, self.max_year, self.min_year)
        df = self.ratio_finance_features(df, self.max_year, self.min_year)

        df = self.cat_one_hot(df, self.cat_cols)

        return df
