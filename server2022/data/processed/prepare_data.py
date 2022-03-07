import pandas as pd
from pathlib import Path


financial_report_columns = [
    '2016, Нематериальные активы, RUB',
    '2017, Нематериальные активы, RUB',
    '2018, Нематериальные активы, RUB',
    '2019, Нематериальные активы, RUB',
    '2020, Нематериальные активы, RUB',
    '2016, Основные средства , RUB',
    '2017, Основные средства , RUB',
    '2018, Основные средства , RUB',
    '2019, Основные средства , RUB',
    '2020, Основные средства , RUB',
    '2016, Внеоборотные активы, RUB',
    '2017, Внеоборотные активы, RUB',
    '2018, Внеоборотные активы, RUB',
    '2019, Внеоборотные активы, RUB',
    '2020, Внеоборотные активы, RUB',
    '2016, Дебиторская задолженность, RUB',
    '2017, Дебиторская задолженность, RUB',
    '2018, Дебиторская задолженность, RUB',
    '2019, Дебиторская задолженность, RUB',
    '2020, Дебиторская задолженность, RUB',
    '2016, Оборотные активы, RUB',
    '2017, Оборотные активы, RUB',
    '2018, Оборотные активы, RUB',
    '2019, Оборотные активы, RUB',
    '2020, Оборотные активы, RUB',
    '2016, Уставный капитал , RUB',
    '2017, Уставный капитал , RUB',
    '2018, Уставный капитал , RUB',
    '2019, Уставный капитал , RUB',
    '2020, Уставный капитал , RUB',
    '2016, Капитал и резервы, RUB',
    '2017, Капитал и резервы, RUB',
    '2018, Капитал и резервы, RUB',
    '2019, Капитал и резервы, RUB',
    '2020, Капитал и резервы, RUB',
    '2016, Заёмные средства (долгосрочные), RUB',
    '2017, Заёмные средства (долгосрочные), RUB',
    '2018, Заёмные средства (долгосрочные), RUB',
    '2019, Заёмные средства (долгосрочные), RUB',
    '2020, Заёмные средства (долгосрочные), RUB',
    '2016, Долгосрочные обязательства, RUB',
    '2017, Долгосрочные обязательства, RUB',
    '2018, Долгосрочные обязательства, RUB',
    '2019, Долгосрочные обязательства, RUB',
    '2020, Долгосрочные обязательства, RUB',
    '2016, Заёмные средства (краткосрочные), RUB',
    '2017, Заёмные средства (краткосрочные), RUB',
    '2018, Заёмные средства (краткосрочные), RUB',
    '2019, Заёмные средства (краткосрочные), RUB',
    '2020, Заёмные средства (краткосрочные), RUB',
    '2016, Кредиторская задолженность, RUB',
    '2017, Кредиторская задолженность, RUB',
    '2018, Кредиторская задолженность, RUB',
    '2019, Кредиторская задолженность, RUB',
    '2020, Кредиторская задолженность, RUB',
    '2016, Краткосрочные обязательства, RUB',
    '2017, Краткосрочные обязательства, RUB',
    '2018, Краткосрочные обязательства, RUB',
    '2019, Краткосрочные обязательства, RUB',
    '2020, Краткосрочные обязательства, RUB',
    '2016, Выручка, RUB',
    '2017, Выручка, RUB',
    '2018, Выручка, RUB',
    '2019, Выручка, RUB',
    '2020, Выручка, RUB',
    '2016, Себестоимость продаж, RUB',
    '2017, Себестоимость продаж, RUB',
    '2018, Себестоимость продаж, RUB',
    '2019, Себестоимость продаж, RUB',
    '2020, Себестоимость продаж, RUB',
    '2016, Прибыль (убыток) до налогообложения , RUB',
    '2017, Прибыль (убыток) до налогообложения , RUB',
    '2018, Прибыль (убыток) до налогообложения , RUB',
    '2019, Прибыль (убыток) до налогообложения , RUB',
    '2020, Прибыль (убыток) до налогообложения , RUB',
    '2016, Прибыль (убыток) от продажи, RUB',
    '2017, Прибыль (убыток) от продажи, RUB',
    '2018, Прибыль (убыток) от продажи, RUB',
    '2019, Прибыль (убыток) от продажи, RUB',
    '2020, Прибыль (убыток) от продажи, RUB'
]

factors_2020 = [f'Факт. {i}' for i in range(1, 61)]

factors_2021 = [
    'Факт. 20',
    'Факт. 21',
    'Факт.32',
    'Факт.31',
    'Факт.23',
    'Факт 24',
    'Факт 27',
    'Факт 33',
    'Факт 28',
    'Факт 29',
    'Факт 30',
    'Факт 40',
    'Факт 41',
    'Факт 42',
    'Факт 46',
    'Факт 48',
    'Факт 49',
    'Факт 50',
    'Факт 51',
    'Факт 54',
    'Факт 55',
    'Факт 56',
    'Факт 57',
    'Факт 58',
    'Факт 59',
    'Факт 60',
    'Факт 37',
    'Факт 39',
    'Факт 1',
    'Факт 2',
    'Факт 3',
    'Факт 7',
    'Факт 12',
    'Факт 14',
    'Факт 15',
    'Факт 16',    
]



#df_2019 = pd.read_csv('./agents2019.csv')
#df_2020 = pd.read_csv('./agents2020.csv')
#df_2021 = pd.read_csv('./agents2021.csv')


def create_df_2years_known():
    file_path = Path(__file__)
    
    df_2_years_known = pd.read_csv(file_path.parent.parent / 'raw/agents2021.csv')
    df_2_years_known.drop(columns='Unnamed: 0', inplace=True)
    df_2_years_known.set_index('Наименование ДП', inplace=True)

    info_2020 = pd.read_csv(file_path.parent.parent / 'raw/agents2020.csv').set_index('Наименование ДП')
    info_2020 = info_2020[[*factors_2020, 'Итого']]
    factors_renaming = {x: x+' (2020)' for x in factors_2020}
    info_2020.rename(columns={**factors_renaming, 'Итого': 'Итого (2020)'}, inplace=True)

    df_2_years_known = df_2_years_known.join(info_2020)
    
    return df_2_years_known



def create_df_0years_known(drop_unnecessary=True, drop_extra_factors=True, drop_2021_unique_feats=True, 
                           drop_5y_ago=True):
    """
    drop_unnecessary: whether to drop targets other than binary PDZ
    """
    dataframes = []
    file_path = Path(__file__)
    
    current_year = 2019
    df_0years_known = pd.read_csv(file_path.parent.parent / 'raw/agents2019.csv')
    df_0years_known['year'] = str(current_year)
    df_0years_known.drop(columns=['Unnamed: 0',], inplace=True)
    financial_report_columns_renaming = {x :f'{int(x[:4]) - current_year}' + x[4:] 
                                         for x in financial_report_columns}
    df_0years_known.rename(columns=financial_report_columns_renaming, inplace=True)
    df_0years_known['binary_target'] = df_0years_known['Кол-во раз ПДЗ за 2019 год, шт.'] != 0
    
    if drop_unnecessary:
        columns_to_drop = [
            'Макс. ПДЗ за 2019 год, дней',
            'Сред. ПДЗ за 2019 год, дней',
            'Кол-во просрочек свыше 5-ти дней за 2019 год, шт.',
            'Общая сумма ПДЗ свыше 5-ти дней за 2019 год, руб.',
            'Кол-во раз ПДЗ за 2019 год, шт.',
        ]
        df_0years_known.drop(columns=columns_to_drop, inplace=True)
    
    dataframes.append(df_0years_known.copy())
        
    
    current_year = 2020
    df_0years_known = pd.read_csv(file_path.parent.parent / 'raw/agents2020.csv')
    df_0years_known['year'] = str(current_year)
    financial_report_columns_renaming = {x :f'{int(x[:4]) - current_year}' + x[4:] 
                                         for x in financial_report_columns}
    df_0years_known.rename(columns=financial_report_columns_renaming, inplace=True)
    df_0years_known['binary_target'] = df_0years_known['Кол-во раз ПДЗ за 2020 год, шт.'] != 0
    
    if drop_unnecessary:
        columns_to_drop = [
            'Макс. ПДЗ за 2020 год, дней',
            'Сред. ПДЗ за 2020 год, дней',
            'Кол-во просрочек свыше 5-ти дней за 2020 год, шт.',
            'Общая сумма ПДЗ свыше 5-ти дней за 2020 год, руб.',
            'Кол-во раз ПДЗ за 2020 год, шт.',
        ]
        df_0years_known.drop(columns=columns_to_drop, inplace=True)
        
    dataframes.append(df_0years_known.copy())
    
    
    current_year = 2021
    df_0years_known = pd.read_csv(file_path.parent.parent / 'raw/agents2021.csv')
    df_0years_known['year'] = str(current_year)

    columns_to_drop = [
        'Unnamed: 0', 
        'Макс. ПДЗ за 2019 год, дней',
        'Сред. ПДЗ за 2019 год, дней',
        'Кол-во просрочек свыше 5-ти дней за 2019 год, шт.',
        'Общая сумма ПДЗ свыше 5-ти дней за 2019 год, руб.',
        'Кол-во раз ПДЗ за 2019 год, шт.',
        'Макс. ПДЗ за 2020 год, дней',
        'Сред. ПДЗ за 2020 год, дней',
        'Кол-во просрочек свыше 5-ти дней за 2020 год, шт.',
        'Общая сумма ПДЗ свыше 5-ти дней за 2020 год, руб.',
        'Кол-во раз ПДЗ за 2020 год, шт.',
    ]
    df_0years_known.drop(columns=columns_to_drop, inplace=True)

    financial_report_columns_renaming = {x :f'{int(x[:4]) - current_year}' + x[4:] 
                                         for x in financial_report_columns}
    df_0years_known.rename(columns=financial_report_columns_renaming, inplace=True)

    factors_nodot = factors_2021[5:]
    factors_renaming = {x: '. '.join(x.split()) for x in factors_nodot}
    factors_renaming.update({
        'Факт.32': 'Факт. 32',   
        'Факт.31': 'Факт. 31',
        'Факт.23': 'Факт. 23'
    })    
    df_0years_known.rename(columns=factors_renaming, inplace=True)
    
    df_0years_known['binary_target'] = df_0years_known[['ПДЗ 1-30', 'ПДЗ 31-90', 'ПДЗ 91-365', 
                                                        'ПДЗ более 365',]].sum(axis=1) > 0
    
    if drop_unnecessary:
        columns_to_drop = [
            'ПДЗ 1-30',
            'ПДЗ 31-90',
            'ПДЗ 91-365',
            'ПДЗ более 365',
        ]
        df_0years_known.drop(columns=columns_to_drop, inplace=True)
    
    if drop_2021_unique_feats:
        columns_to_drop = [
            'Оценка потенциала контрагента 1, руб.',
            'Оценка потенциала контрагента 2, руб.',
            'Статус',
        ]
        df_0years_known.drop(columns=columns_to_drop, inplace=True)
    
    dataframes.append(df_0years_known.copy())
    
    result = pd.concat(dataframes, axis=0).reset_index(drop=True)
    
    if drop_extra_factors:
        usefull_factors = factors_2021[:2] + list(factors_renaming.values())
        extra_factors = list(set(factors_2020) - set(usefull_factors))
        result.drop(columns=extra_factors, inplace=True)

    if drop_5y_ago:
        cols = result.columns.tolist()
        cols_5y_ago = [x for x in cols if x.startswith('-5')]
        result.drop(columns=cols_5y_ago, inplace=True)
        
    return result



def stats_PDZ_names(year):
    names = [
        'Макс. ПДЗ за {} год, дней',
        'Сред. ПДЗ за {} год, дней',
        'Кол-во просрочек свыше 5-ти дней за {} год, шт.',
        'Общая сумма ПДЗ свыше 5-ти дней за {} год, руб.',
        'Кол-во раз ПДЗ за {} год, шт.',
    ]
    return [x.format(year) for x in names], {x.format(year): x.format(-1) for x in names}


def create_df_1year_known_2020(drop_unnecessary=True):
    current_year = 2020
    file_path = Path(__file__)
    df_ = pd.read_csv(file_path.parent.parent / 'raw/agents2020.csv')
    df_['year'] = str(current_year)
    financial_report_columns_renaming = {x :f'{int(x[:4]) - current_year}' + x[4:] 
                                         for x in financial_report_columns}
    df_.rename(columns=financial_report_columns_renaming, inplace=True)
    df_['binary_target'] = df_['Кол-во раз ПДЗ за 2020 год, шт.'] != 0

    if drop_unnecessary:
        columns_to_drop = stats_PDZ_names(current_year)[0]
        df_.drop(columns=columns_to_drop, inplace=True)

    df_.set_index('Наименование ДП', inplace=True)
    cols, cols_renaming = stats_PDZ_names(current_year-1)
    df_prev = pd.read_csv(file_path.parent.parent / 'raw/agents2019.csv').set_index('Наименование ДП')[cols]
    
    df_ = df_.join(df_prev)
    df_.rename(columns=cols_renaming, inplace=True)
    
    return df_.reset_index()


def create_df_1year_known_2021(drop_unnecessary=True):
    current_year = 2021
    file_path = Path(__file__)
    df_ = pd.read_csv(file_path.parent.parent / 'raw/agents2021.csv')
    df_['year'] = str(current_year)
    financial_report_columns_renaming = {x :f'{int(x[:4]) - current_year}' + x[4:] 
                                         for x in financial_report_columns}
    df_.rename(columns=financial_report_columns_renaming, inplace=True)
    df_['binary_target'] = df_['Кол-во раз ПДЗ за 2020 год, шт.'] != 0

    columns_to_drop = [
        'Unnamed: 0', 
        'Макс. ПДЗ за 2019 год, дней',
        'Сред. ПДЗ за 2019 год, дней',
        'Кол-во просрочек свыше 5-ти дней за 2019 год, шт.',
        'Общая сумма ПДЗ свыше 5-ти дней за 2019 год, руб.',
        'Кол-во раз ПДЗ за 2019 год, шт.',
    ]
    df_.drop(columns=columns_to_drop, inplace=True)
    df_.rename(columns=stats_PDZ_names(current_year-1)[1], inplace=True)

    factors_nodot = factors_2021[5:]
    factors_renaming = {x: '. '.join(x.split()) for x in factors_nodot}
    factors_renaming.update({
        'Факт.32': 'Факт. 32',   
        'Факт.31': 'Факт. 31',
        'Факт.23': 'Факт. 23'
    })
    df_.rename(columns=factors_renaming, inplace=True)

    df_['binary_target'] = df_[['ПДЗ 1-30', 'ПДЗ 31-90', 'ПДЗ 91-365', 
                                'ПДЗ более 365',]].sum(axis=1) > 0

    if drop_unnecessary:
        columns_to_drop = [
            'ПДЗ 1-30',
            'ПДЗ 31-90',
            'ПДЗ 91-365',
            'ПДЗ более 365',
        ]
        df_.drop(columns=columns_to_drop, inplace=True)

    df_.set_index('Наименование ДП', inplace=True)
    cols = factors_2020 + ['Итого',]
    df_prev = pd.read_csv(file_path.parent.parent / 'raw/agents2020.csv').set_index('Наименование ДП')[cols]
    df_prev.rename(columns={x: x + ' (-1)' for x in cols}, inplace=True)
    df_ = df_.join(df_prev)

    return df_.reset_index()


def create_df_1year_known(drop_unnecessary=True):
    df_2020 = create_df_1year_known_2020(drop_unnecessary=drop_unnecessary)
    df_2021 = create_df_1year_known_2021(drop_unnecessary=drop_unnecessary)
    
    return pd.concat([df_2020, df_2021], axis=0).reset_index(drop=True)
