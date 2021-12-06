
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Date
import datetime
import os
import sqlite3
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


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
                         'Przypis': Integer,
                         'TU Inkaso': Integer,
                         },
                  if_exists='replace')

        return engine


def zapisz_excel(df, output):
    df.to_excel(output, index=False)


def msc():
    return ['styczeń', 'luty', 'marzec',
            'kwiecień', 'maj', 'czerwiec',
            'lipiec', 'sierpień', 'wrzesień',
            'październik', 'listopad', 'grudzień']


def inkaso_agencji(df, msc):
    df = df[df['TUrozlcz?'] == 'rozl']
    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)
    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum().reset_index()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1, marker='o')

    time_len = range(len(dff['Strzałka czasu']))
    model = np.polyfit(time_len, dff['Inkaso w PLN --> przychód'], 1)
    predict = np.poly1d(model)

    plt.plot(time_len, predict(time_len), ls="--")

    ax.grid(which='major', color='black', linewidth=0.075)

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYCHÓD AGENCJI (Składki opłacone)')
    plt.legend(['przychód'])
    plt.show()


def przypis_inkaso_agencji(df, msc):
    df = df[df['TUrozlcz?'] == 'rozl']
    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)
    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum().reset_index()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 14)});fig, ax = plt.subplots();fig.autofmt_xdate()

    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1, marker='o')
    ax = sns.lineplot(x='Strzałka czasu', y='Przypis', data=dff, lw=1, marker='o')
    ax.grid(which='major', color='black', linewidth=0.075)

    time_len = range(len(dff['Strzałka czasu']))

    model_inkaso = np.polyfit(time_len, dff['Inkaso w PLN --> przychód'], 1)
    predict_inkaso = np.poly1d(model_inkaso)
    plt.plot(time_len, predict_inkaso(time_len), ls="--")

    model_przypis = np.polyfit(time_len, dff['Przypis'], 1)
    predict_przypis = np.poly1d(model_przypis)
    plt.plot(time_len, predict_przypis(time_len), ls="--")

    print('r2: ', r2_score(time_len, predict_przypis(time_len)))

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYPIS i INKASO AGENCJI (Składki opłacone)')
    plt.legend(['przychód', 'przypis'])
    plt.show()


def inkaso_magro(df, msc):
    df = df[(df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) & (df['TUrozlcz?'] == 'rozl')]

    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)

    # Tylko Robert
    df = df.append({'Strzałka czasu': '1707',
                    'Inkaso w PLN --> przychód': 0},
                   ignore_index=True)
    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum().reset_index()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1, marker='o')

    time_len = range(len(dff['Strzałka czasu']))
    model = np.polyfit(time_len, dff['Inkaso w PLN --> przychód'], 1)
    predict = np.poly1d(model)

    plt.plot(time_len, predict(time_len), ls="--")

    ax.grid(which='major', color='black', linewidth=0.075)

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYCHÓD MAGRO -> Maciek, Robert. (Składki opłacone)')
    plt.legend(['przychód'])
    plt.show()


def przypis_inkaso_magro(df, msc):
    df = df[(df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) & (df['TUrozlcz?'] == 'rozl')]

    df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
    df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))
    df.insert(3, 'Strzałka czasu', df_msc)

    # Tylko Robert
    df = df.append({'Strzałka czasu': '1702',
                    'Inkaso w PLN --> przychód': 0},
                   ignore_index=True)
    df = df.append({'Strzałka czasu': '1707',
                    'Inkaso w PLN --> przychód': 0},
                   ignore_index=True)

    df = df.sort_values(by=['Strzałka czasu'])
    dff = df.groupby(['Strzałka czasu']).sum().reset_index()
    rok_msc = df['Strzałka czasu'].unique()
    rok = [rok[:2] for rok in rok_msc if rok is not None]

    sns.set(rc={'figure.figsize': (29, 14)});fig, ax = plt.subplots();fig.autofmt_xdate()

    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1, marker='o')
    ax = sns.lineplot(x='Strzałka czasu', y='Przypis', data=dff, lw=1, marker='o')
    ax.grid(which='major', color='black', linewidth=0.075)

    time_len = range(len(dff['Strzałka czasu']))

    model_inkaso = np.polyfit(time_len, dff['Inkaso w PLN --> przychód'], 1)
    predict_inkaso = np.poly1d(model_inkaso)
    plt.plot(time_len, predict_inkaso(time_len), ls="--")

    model_przypis = np.polyfit(time_len, dff['Przypis'], 1)
    predict_przypis = np.poly1d(model_przypis)
    plt.plot(time_len, predict_przypis(time_len), ls="--")

    print('r2: ', r2_score(time_len, predict_przypis(time_len)))

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYPIS i PRZYCHÓD MAGRO -> Maciek, Robert. (Składki opłacone)')
    plt.legend(['przychód', 'przypis'])
    plt.show()


if __name__ == '__main__':
    excel = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    output = 'output.xlsx'

    engine = make_sql(excel)
    sql_df = pd.read_sql('Select * from baza', engine)

    # zapisz_excel(sql_df, output)

    msc = msc()

    inkaso_agencji(sql_df, msc)
    przypis_inkaso_agencji(sql_df, msc)
    inkaso_magro(sql_df, msc)
    przypis_inkaso_magro(sql_df, msc)