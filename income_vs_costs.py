import os
from sqlalchemy import create_engine
import pandas as pd
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
                      header=0)
    return csv


if __name__ == '__main__':

    # file = "/run/user/1000/gvfs/smb-share:server=192.168.1.12,share=e/Agent baza/2014 BAZA MAGRO.xlsx"
    file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    bank_file = '/home/robb/Desktop/historia_2021-12-21_20109027050000000133736204.csv'

    engine = make_sql(file)

    sql_df = pd.read_sql('Select * from baza', engine)

    bank = read_bank(bank_file)

    bank_data = pd.DataFrame(bank)

    print(bank_data)

    # print(sql_df)

