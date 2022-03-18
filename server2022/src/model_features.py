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
        
        self.abs_feat = ['-4, Нематериальные активы, RUB',
       '-3, Нематериальные активы, RUB', '-2, Нематериальные активы, RUB',
       '-1, Нематериальные активы, RUB', '-4, Основные средства , RUB',
       '-3, Основные средства , RUB', '-2, Основные средства , RUB',
       '-1, Основные средства , RUB', '-4, Внеоборотные активы, RUB',
       '-3, Внеоборотные активы, RUB', '-2, Внеоборотные активы, RUB',
       '-1, Внеоборотные активы, RUB', '-4, Дебиторская задолженность, RUB',
       '-3, Дебиторская задолженность, RUB',
       '-2, Дебиторская задолженность, RUB',
       '-1, Дебиторская задолженность, RUB', '-4, Оборотные активы, RUB',
       '-3, Оборотные активы, RUB', '-2, Оборотные активы, RUB',
       '-1, Оборотные активы, RUB', '-4, Уставный капитал , RUB',
       '-3, Уставный капитал , RUB', '-2, Уставный капитал , RUB',
       '-1, Уставный капитал , RUB', '-4, Капитал и резервы, RUB',
       '-3, Капитал и резервы, RUB', '-2, Капитал и резервы, RUB',
       '-1, Капитал и резервы, RUB',
       '-4, Заёмные средства (долгосрочные), RUB',
       '-3, Заёмные средства (долгосрочные), RUB',
       '-2, Заёмные средства (долгосрочные), RUB',
       '-1, Заёмные средства (долгосрочные), RUB',
       '-4, Долгосрочные обязательства, RUB',
       '-3, Долгосрочные обязательства, RUB',
       '-2, Долгосрочные обязательства, RUB',
       '-1, Долгосрочные обязательства, RUB',
       '-4, Заёмные средства (краткосрочные), RUB',
       '-3, Заёмные средства (краткосрочные), RUB',
       '-2, Заёмные средства (краткосрочные), RUB',
       '-1, Заёмные средства (краткосрочные), RUB',
       '-4, Кредиторская задолженность, RUB',
       '-3, Кредиторская задолженность, RUB',
       '-2, Кредиторская задолженность, RUB',
       '-1, Кредиторская задолженность, RUB',
       '-4, Краткосрочные обязательства, RUB',
       '-3, Краткосрочные обязательства, RUB',
       '-2, Краткосрочные обязательства, RUB',
       '-1, Краткосрочные обязательства, RUB', '-4, Выручка, RUB',
       '-3, Выручка, RUB', '-2, Выручка, RUB', '-1, Выручка, RUB',
       '-4, Себестоимость продаж, RUB', '-3, Себестоимость продаж, RUB',
       '-2, Себестоимость продаж, RUB', '-1, Себестоимость продаж, RUB',
       '-4, Прибыль (убыток) до налогообложения , RUB',
       '-3, Прибыль (убыток) до налогообложения , RUB',
       '-2, Прибыль (убыток) до налогообложения , RUB',
       '-1, Прибыль (убыток) до налогообложения , RUB',
       '-4, Прибыль (убыток) от продажи, RUB',
       '-3, Прибыль (убыток) от продажи, RUB',
       '-2, Прибыль (убыток) от продажи, RUB',
       '-1, Прибыль (убыток) от продажи, RUB',]

        self.cat_cols = ["Наименование ДП"]

        self.max_lookback = max_lookback
        self.min_lookback = min_lookback

    def get_full_finance_feat_name(self, fin_feat, i):
        return str(i) + ', ' + fin_feat + ', RUB'

    def diff_finance_features(self, df, max_lookback, min_lookback, use_bins=False):
        for fin_feat in self.finance_feat:
            current_cols = [fin_feat + f"_,прирост_за_{year + 1}_год" for year in
                            range(max_lookback, min_lookback)]
            for year in range(max_lookback, min_lookback):
                df[fin_feat + f"_,прирост_за_{year + 1}_год"] = (
                    df[self.get_full_finance_feat_name(fin_feat, year + 1)]
                    - df[self.get_full_finance_feat_name(fin_feat, year)]
                )

#                 scaler = MinMaxScaler()
#                 df[fin_feat + f" ,прирост за {year + 1} год"] = scaler.fit_transform(
#                     df[fin_feat + f" ,прирост за {year + 1} год"].values.reshape(-1, 1)
#                 )
            if use_bins:
                for i in range(df.shape[0]):
                    for col in current_cols:
                        q1, q2, q3 = self.get_quantiles(df, col)
                        df.loc[i, col] = self.get_bin_label(df.loc[i, col], q1, q2, q3)

        return df


    def ratio_finance_features(self, df, max_lookback, min_lookback):
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
                    if np.isnan(df.loc[i, col]):
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
        return [x for x in df.columns if 'Факт' in x]
    
    def get_quantiles(self, df, column):
        #q_25, q_50, q_75 = df[column].values.mean() * 0.25, df[column].values.mean() * 0.5, df[column].values.mean() * 0.75
        q_25, q_50, q_75 = np.quantile(df[column].values, 0.25), np.quantile(df[column].values, 0.5), np.quantile(df[column].values, 0.75)
        return q_25, q_50, q_75
    
    def get_bin_label(self, value, q1, q2, q3):
        if value <= q1:
            return 0
        elif value <= q2:
            return 1
        elif value <= q3:
            return 2
        else:
            return 3 
        
    def bins(self, df, column):
        return pd.Series(self.get_bin_label(row, column, self.get_quantiles(df, column)) for row in df.itertuples())
    

    def preprocessing_before_fitting(self, df, use_diff_features=True, use_ratio_features=True, use_bins=False):
        if use_diff_features:
            df = self.diff_finance_features(df, self.max_lookback, self.min_lookback, use_bins)
        if use_ratio_features:
            df = self.ratio_finance_features(df, self.max_lookback, self.min_lookback)

        #df = self.scaling(df)
        if use_bins:
            for i in range(df.shape[0]):
                    for col in self.abs_feat:
                        q1, q2, q3 = self.get_quantiles(df, col)
                        df.loc[i, col] = self.get_bin_label(df.loc[i, col], q1, q2, q3)

        cat_col = self.get_cat_feat_name(df)
        other_col = [x for x in df.columns if x not in cat_col]

        df.fillna(0, inplace=True)

        return df[other_col]