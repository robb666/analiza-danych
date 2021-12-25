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
    november_2021 = 15053  # cofnicie sie do oplaconych skladek
    df_magro = db[
                  (db['index'] > april_2020) &
                  (db['index'] < november_2021)
    ]

    # df_magro = db
    # df_bank = pd.DataFrame(bank)

    # print(df_magro.index)
    # print(df_bank.index.astype('Int64'))


    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    # plt.gca().set(xlim=(0, 15249))
    # print(df_magro.index)

    # ax = sns.regplot(x=df_magro.index, y=df_magro.Przypis,
    #                  scatter=None,
    #                  order=2,
    #                  scatter_kws={'s': 10, 'alpha': 0.4},
    #                  line_kws={'lw': 1, 'color': 'g'})


    # print(df_magro.head())

    df_magro2 = df_magro[['Data wystawienia', 'TU Inkaso']]

    # print(df_magro2['Data wystawienia'])
    df_magro2['Data nowa'] = df_magro2['Data wystawienia'].fillna('2021-04-27')
    df_magro2['Data nowa'] = df_magro2['Data nowa'].apply(lambda x: x[:-3])

    df_magro2['Data nowa'] = pd.to_datetime(df_magro2['Data nowa'])
    df_magro2 = df_magro2.groupby(['Data nowa']).sum().reset_index()
    df_magro2 = df_magro2[3:-1]
    df_magro2['TU Inkaso'] = df_magro2['TU Inkaso'].astype(int)
    x = df_magro2['Data nowa']
    print(df_magro2)
    ax.xaxis.update_units(x)

    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='TU Inkaso',
                     # scatter=None,
                     data=df_magro2,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4},
                     line_kws={'lw': 1, 'color': 'black'})


    # print(df_bank)
    df_bank.Kwota = df_bank.Kwota.replace({',': '.'}, regex=True)
    df_bank.Kwota = df_bank.Kwota.astype(float) * -1

    df_bank['Data nowa'] = df_bank['Data księgowania'].apply(lambda x: x[3:])

    df_bank = df_bank[['Data nowa', 'Kwota']]
    df_bank = df_bank.groupby(['Data nowa']).sum().reset_index()

    df_bank.Kwota = df_bank.Kwota.astype(int)

    df_srt = df_bank.sort_values(by=['Data nowa'])
    df_srt['Data nowa'] = pd.to_datetime(df_srt['Data nowa'])
    df_bank2 = df_srt.sort_values('Data nowa')

    x = df_bank2['Data nowa']
    # print(df_bank2)

    ax.xaxis.update_units(x)

    print(df_bank2.dtypes)
    # ax = sns.regplot(x='Data nowa',
    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='Kwota',
                     data=df_bank2,
                     # scatter=None,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4},
                     line_kws={'lw': 1, 'color': 'g'})

    # ax.set_xticklabels(df_bank['Data księgowania'], rotation=45)
    plt.show()



if __name__ == '__main__':

    # file = "/run/user/1000/gvfs/smb-share:server=192.168.1.12,share=e/Agent baza/2014 BAZA MAGRO.xlsx"
    db_file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    bank_statement = '/home/robb/Desktop/historia_2021-12-25_20109027050000000133736204.csv'

    engine = make_sql(db_file)

    sql_df = pd.read_sql('Select * from baza', engine)

    bank = read_bank(bank_statement)


    print(plot(sql_df, bank))


