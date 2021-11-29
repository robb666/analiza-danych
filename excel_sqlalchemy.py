from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import Date
import datetime
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


Base = declarative_base()

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


def baza_sql(file, engine):
    df = pd.read_excel(file, sheet_name='BAZA 2014', header=1)
    new_header = df.iloc[0]
    df.columns = new_header
    df = df[2:]
    df.reset_index()

    df.to_sql('baza', engine,
              dtype={
                     'Data wystawienia': Date,
                     'Przypis': Integer,
                     'TU Inkaso': Integer,
                     },
              if_exists='replace')

    # return pd.read_sql('Select * from baza', engine)


def zapisz_excel(baza, output):
    baza.to_excel(output, index=False)


def msc():
    return ['styczeń', 'luty', 'marzec',
            'kwiecień', 'maj', 'czerwiec',
            'lipiec', 'sierpień', 'wrzesień',
            'październik', 'listopad', 'grudzień']


def inkaso(engine, msc):
    if not engine:
        raise AssertionError

    df = pd.read_sql('Select * from baza', engine)
    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)
    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1)

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('INKASO AGENCJI')
    plt.show()


def przypis_inkaso(engine, msc):
    df = pd.read_sql('Select * from baza', engine)
    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)
    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1)
    ax = sns.lineplot(x='Strzałka czasu', y='Przypis', data=dff, lw=1)

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYPIS i INKASO AGENCJI')
    plt.show()


if __name__ == '__main__':
    file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    output = 'output.xlsx'

    # engine = create_engine('sqlite:///baza_sql.db', echo=False)

    # read_sql = baza_sql(file, engine)
    msc = msc()

    inkaso(engine, msc)
    przypis_inkaso(engine, msc)
