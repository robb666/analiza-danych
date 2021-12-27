import os
import numpy as np
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import Date
from sqlalchemy import Integer
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def make_sql(file):

    db_path = '/home/robb/Desktop/PROJEKTY/analiza-danych/baza_sql.db'

    if os.path.exists(db_path):
        return 'sqlite:///baza_sql.db'
    else:
        engine = create_engine('sqlite:///baza_sql.db', echo=False)
        df = pd.read_excel(file, sheet_name='BAZA 2014', header=1)
        new_header = df.iloc[0]
        df.columns = new_header
        df = df[2:]
        df.reset_index()

        df.to_sql('baza', engine,
                  dtype={
                         'Data wystawienia': Date,
                         'Początek': Date,
                         'Koniec': Date,
                         'Przypis': Integer,
                         'Data rat': Date,
                         'Data inkasa': Date,
                         'TU Inkaso': Integer,
                         'TU Raty': Integer,
                         },
                  if_exists='append')

        return engine


def read_bank(file):
    csv = pd.read_csv(file,
                      names=[
                             'Data księgowania',
                             'Data zlecenia',
                             'Tytuł',
                             'Nadawca',
                             'Nr rachunku',
                             'Kwota',
                             'Saldo',
                             'idx',
                             'nanana'
                             ],
                      usecols=['Data księgowania', 'Tytuł', 'Nadawca', 'Kwota', 'Saldo'],
                      header=0).sort_index(axis=0, ascending=False, ignore_index=True)
    return csv


def plot(db, df_bank):
    april_2020 = 10612  # data od czasu rozliczeń tylko na Spółkę
    december_2020 = 12555  # rok 2021
    november_2021 = 15053  # cofnicie sie do oplaconych skladek
    df_income = db[
                  (db['index'] > december_2020) &
                  ~(db['Miesiąc przypisu'] == '20_12') &
                  (db['index'] < november_2021)
    ]
    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    # plt.gca().set(xlim=(0, 15249))
    # print(df_magro.index)

    # ax = sns.regplot(x=df_magro.index,
    #                  y=df_magro.Przypis,
    #                  scatter=None,
    #                  order=2,
    #                  scatter_kws={'s': 10, 'alpha': 0.4},
    #                  line_kws={'lw': 1, 'color': 'g'})

    # print(df_magro.head())

    df_income2 = df_income[['Data wystawienia', 'TU Inkaso']]
    df_income2['Data nowa'] = df_income2['Data wystawienia'].fillna('2021-04-27')
    # df_magro2.loc[666:, ('Data nowa')] = '2021-04-27'
    df_income2['Data nowa'] = df_income2['Data nowa'].apply(lambda x: x[:-3])

    df_income2['Data nowa'] = pd.to_datetime(df_income2['Data nowa'])
    df_income2 = df_income2.groupby(['Data nowa']).sum().reset_index()

    df_income2 = df_income2[(df_income2['Data nowa'] > pd.to_datetime('2020-12')) &
                          (df_income2['Data nowa'] < pd.to_datetime('2021-11'))]

    df_income2['TU Inkaso'] = df_income2['TU Inkaso'].astype(int)
    print(df_income2)
    x = df_income2['Data nowa']
    # print(df_magro2)
    print(f"\nSuma Inkasa: {df_income2['TU Inkaso'].sum()} zł")
    ax.xaxis.update_units(x)

    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='TU Inkaso',
                     # scatter=None,
                     data=df_income2,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4},
                     line_kws={'lw': 1, 'color': 'g'})


    df_bank.Kwota = df_bank.Kwota.replace({',': '.'}, regex=True)
    df_bank.Kwota = df_bank.Kwota.astype(float)

    df_bank['Data nowa'] = df_bank['Data księgowania'].apply(lambda x: x[3:])

    df_costs = df_bank[['Data nowa', 'Nadawca', 'Kwota']]
    df_costs['Nadawca'] = df_costs['Nadawca'].fillna('bez tyt.')
    mg = df_costs[df_costs['Nadawca'].str.contains('MAGRO MACIEJ')]
    mg.Kwota = mg.Kwota * -1

    df_costs = df_costs[df_costs['Kwota'] < 0]
    df_costs.Kwota = df_costs.Kwota * -1

    df_costs = pd.concat([df_costs, mg], axis=1, ignore_index=False)
    df_costs = df_costs.groupby(df_costs.columns, axis=1).sum()

    df_costs = df_costs.groupby(['Data nowa']).sum().reset_index()
    df_costs.Kwota = df_costs.Kwota.astype(int)

    df_costs['Data nowa'] = pd.to_datetime(df_costs['Data nowa'])
    df_costs = df_costs.sort_values('Data nowa')
    df_costs = df_costs[df_costs['Data nowa'] > pd.to_datetime('2020-12')]
    print(df_costs)
    x = df_costs['Data nowa']
    ax.xaxis.update_units(x)

    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='Kwota',
                     data=df_costs,
                     # scatter=None,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4},
                     line_kws={'lw': 1, 'color': 'r'})

    # ax.set_xticklabels(df_bank['Data księgowania'], rotation=45)
    plt.show()


if __name__ == '__main__':

    # file = "/run/user/1000/gvfs/smb-share:server=192.168.1.12,share=e/Agent baza/2014 BAZA MAGRO.xlsx"
    db_file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    bank_statement = '/home/robb/Desktop/historia_2021-12-26_20109027050000000133736204.csv'

    engine = make_sql(db_file)

    sql_df = pd.read_sql('Select * from baza', engine)

    bank = read_bank(bank_statement)

    print(plot(sql_df, bank))
