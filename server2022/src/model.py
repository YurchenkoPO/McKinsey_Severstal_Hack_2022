import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Feature_gen:
    def __init__(self, max_lookback, min_lookback):
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

        self.max_lookback = max_lookback
        self.min_lookback = min_lookback

    def get_full_finance_feat_name(self, fin_feat, i):
        return str(i) + ', ' + fin_feat + ', RUB'

    def diff_finance_features(self, df, max_lookback, min_lookback):
        for fin_feat in self.finance_feat:
            for year in range(max_lookback, min_lookback):
                df[fin_feat + f" ,прирост за {year + 1} год"] = (
                    df[self.get_full_finance_feat_name(fin_feat, year + 1)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                )

                scaler = MinMaxScaler()
                df[fin_feat + f" ,прирост за {year + 1} год"] = scaler.fit_transform(
                    df[fin_feat + f" ,прирост за {year + 1} год"].values.reshape(-1, 1)
                )
        return df

    def ratio_finance_features(self, df, max_lookback, min_lookback):
        for fin_feat in self.finance_feat:
            for year in range(max_lookback, min_lookback):
                df[fin_feat + f", относительный прирост за {year} год"] = (
                    df[self.get_full_finance_feat_name(fin_feat, year + 1)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                ) / df[self.get_full_finance_feat_name(fin_feat, year)]
                df[fin_feat + f", относительный прирост за {year} год"].fillna(0, inplace=True)

                df.replace([np.inf], 1, inplace=True)
                df.replace([-np.inf], -1, inplace=True)

                scaler = MinMaxScaler()
                df[fin_feat + f", относительный прирост за {year} год"] = scaler.fit_transform(
                    df[fin_feat + f", относительный прирост за {year} год"].values.reshape(-1, 1)
                )

                df[fin_feat + f", прирост относительно выручки за {year} год"] = (
                    df[self.get_full_finance_feat_name(fin_feat, year + 1)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                ) / df[self.get_full_finance_feat_name('Выручка', year)]
                df[fin_feat + f", прирост относительно выручки за {year} год"].fillna(0, inplace=True)

                df.replace([np.inf], 1, inplace=True)
                df.replace([-np.inf], -1, inplace=True)

                scaler = MinMaxScaler()
                df[fin_feat + f", прирост относительно выручки за {year} год"] = scaler.fit_transform(
                    df[fin_feat + f", прирост относительно выручки за {year} год"].values.reshape(-1, 1)
                )

        return df

    def ratio_finance_features_2(self, df, max_lookback, min_lookback):
        for fin_feat in self.finance_feat:
            current_cols = [fin_feat + f", относительный прирост за {year + 1} год" for year in
                            range(max_lookback, min_lookback)]
            for year in range(max_lookback, min_lookback):
                df[fin_feat + f", относительный прирост за {year + 1} год"] = \
                    (df[self.get_full_finance_feat_name(fin_feat, year + 1)] - df[
                        self.get_full_finance_feat_name(fin_feat, year)]) \
                    / df[self.get_full_finance_feat_name(fin_feat, year)]
                # df[fin_feat + f", относительный прирост за {year} год"].fillna(0, inplace=True)

                df.replace([np.inf], np.nan, inplace=True)
                df.replace([-np.inf], np.nan, inplace=True)

            for i in range(df.shape[0]):
                for col in current_cols:
                    df.loc[i, col] = df.iloc[i][current_cols].mean()

        return df

    def scaling(self, df):
        for fin_feat in self.finance_feat:
            for year in range(self.max_lookback, self.min_lookback + 1):
                scaler = MinMaxScaler()
                df[self.get_full_finance_feat_name(fin_feat, year)] = scaler.fit_transform(
                    df[self.get_full_finance_feat_name(fin_feat, year)].values.reshape(-1, 1)
                )

        return df

    def cat_one_hot(self, df, cat_cols):
        df[cat_cols].fillna(str(0), inplace=True)
        df = pd.get_dummies(df, columns=cat_cols)

        return df

    def get_cat_feat_name(self, df):
        return [x for x in df.columns if 'Факт' in x] + ['Итого']

    def preprocessing_before_fitting(self, df):
        df = self.diff_finance_features(df, self.max_lookback, self.min_lookback)
        df = self.ratio_finance_features_2(df, self.max_lookback, self.min_lookback)

        df = self.scaling(df)

        cat_col = self.get_cat_feat_name(df)
        other_col = [x for x in df.columns if x not in cat_col]

        df = self.cat_one_hot(df, cat_col)
        df.fillna(0, inplace=True)

        return df
