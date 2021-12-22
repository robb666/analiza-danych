import os
from sqlalchemy import create_engine
import pandas as pd
from sqlalchemy import Date
from sqlalchemy import Integer
import matplotlib.pyplot as plt
import seaborn as sns


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
    df_magro = db[db['index'] > april_2020].head()


    print(df_bank)
    ax = sns.scatterplot(x='Saldo', data=df_bank)

    # ax.set_xticklabels(df_bank['Data księgowania'], rotation=45)
    # plt.show()

    return df_bank


if __name__ == '__main__':

    # file = "/run/user/1000/gvfs/smb-share:server=192.168.1.12,share=e/Agent baza/2014 BAZA MAGRO.xlsx"
    db_file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    bank_statement = '/home/robb/Desktop/historia_2021-12-21_20109027050000000133736204.csv'

    engine = make_sql(db_file)

    sql_df = pd.read_sql('Select * from baza', engine)

    bank = read_bank(bank_statement)


    print(plot(sql_df, bank))


