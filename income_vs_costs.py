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


    # ax = sns.regplot(df_magro.index, df_magro['TU Inkaso'],
    #                  scatter=None,
    #                  order=2,
    #                  scatter_kws={'s': 10, 'alpha': 0.4},
    #                  line_kws={'lw': 1, 'color': 'black'})



    x = np.arange(len(df_bank['Data księgowania']))
    df_bank.insert(0, 'index', x)
    df_bank.index = df_bank.index.astype(int)
    df_bank.Saldo = df_bank.Saldo.apply(lambda x: str(x)[:-3]).astype(int)


    df_bank['Data nowa'] = df_bank['Data księgowania'].apply(lambda x: x[3:])

    # df_bank['Data księgowania'] = df_bank['Data księgowania']#.map(lambda x: pd.to_datetime(x))#.strftime('%d-%m-%Y'))

    # df_bankv = ax.xaxis.update_units(df_bank['Data księgowania'])
    new_df = df_bank[['Data nowa', 'Saldo']]
    df_bankv = new_df.groupby(['Data nowa']).sum()
    df_bankv = df_bankv.sort_values(by=['Data nowa']).reset_index()
    print(df_bankv)

    # print(df_bank.dtypes)

    ax = sns.regplot(x='Data nowa',
    # ax = sns.regplot(x='Data księgowania',
                     y='Saldo',
                     data=df_bankv,
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


