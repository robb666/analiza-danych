from typing import Tuple, Any
from numpy import ndarray
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy import Date
from sqlalchemy import DateTime
import datetime
from datetime import timedelta
import os
import sqlite3
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates
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
    ax.set_title('PRZYCHÓD AGENCJI (Składki zainkasowane)')
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

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYPIS i INKASO AGENCJI (Składki zainkasowane)')
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

    print('r2 inkaso: ', r2_score(dff['Inkaso w PLN --> przychód'], predict(time_len)))

    plt.plot(time_len, predict(time_len), ls="--")

    ax.grid(which='major', color='black', linewidth=0.075)

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYCHÓD MAGRO -> Maciek, Robert. (Składki zainkasowane)')
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

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()

    ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, lw=1, marker='o')
    ax = sns.lineplot(x='Strzałka czasu', y='Przypis', data=dff, lw=1, marker='o')
    ax.grid(which='major', color='black', linewidth=0.075)

    time_len = range(len(dff['Strzałka czasu']))

    model_inkaso: tuple[ndarray, Any, ndarray] = np.polyfit(time_len, dff['Inkaso w PLN --> przychód'], 1)
    predict_inkaso = np.poly1d(model_inkaso)
    plt.plot(time_len, predict_inkaso(time_len), ls="--")

    print('r2 inkaso: ', r2_score(dff['Inkaso w PLN --> przychód'], predict_inkaso(time_len)))

    time_len = range(len(dff['Strzałka czasu']))
    model_przypis = np.polyfit(time_len, dff['Przypis'], 1)
    predict_przypis = np.poly1d(model_przypis)
    plt.plot(time_len, predict_przypis(time_len), ls="--")

    print('r2 przypis: ', r2_score(dff['Przypis'], predict_przypis(time_len)))

    ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
    ax.set_title('PRZYPIS i PRZYCHÓD MAGRO -> Maciek, Robert. (Składki zainkasowane)')
    plt.legend(['przychód', 'przypis'])
    plt.show()


def displot_przypis(df):
    df = df[(df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) & (df['TUrozlcz?'] == 'rozl')]
    ax = sns.displot(df['Przypis'], height=12.5)
    plt.ticklabel_format(style='plain', axis='x')
    plt.show()


def displot_rocznik(df):
    df = df[(df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) &
            (df['TUrozlcz?'] == 'rozl') &
            (df['Nr rej miejscowość ulica nr'].str.len() < 9) &
            ~(df['Nr rej miejscowość ulica nr'].str.contains('[a-z]', na=False)) &
            (df['Rok produkcji'].str.len() == 4)]

    df = df.sort_values(by='Rok produkcji')

    x = df['Rok produkcji']

    ax = sns.displot(x, kde=True, height=12.5)
    ax.set_xticklabels(rotation=40)

    x_int = df['Rok produkcji'].astype(int)
    mean = x_int.mean()
    ax = sns.displot(x_int, kde=True, height=12.5)
    plt.axvline(mean, 0, 400, color='red')
    ax.set_xticklabels(rotation=40)
    plt.show()


def rocznik_przypis(df):
    """Relacja między rocznikiem auta a przypisem."""
    df = df[(df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) &
            (df['TUrozlcz?'] == 'rozl') &
            (df['Nr rej miejscowość ulica nr'].str.len() < 9) &
            ~(df['Nr rej miejscowość ulica nr'].str.contains('[a-z]', na=False)) &
            (df['Rok produkcji'].str.len() == 4) &
            (df['Przypis'] > 0)]

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()

    df = df.sort_values(by='Rok produkcji')

    df['Rok produkcji'] = df['Rok produkcji'].astype(int)
    ax = sns.scatterplot(x='Rok produkcji', y='Przypis', data=df)

    slope2, slope, intercept = np.polyfit(df['Rok produkcji'], df['Przypis'], 2)
    line_values = [slope2 * i ** 2 + slope * i + intercept for i in df['Rok produkcji']]  # prediction

    print(r2_score(df['Przypis'], line_values))

    plt.plot(df['Rok produkcji'], line_values, ls="--", c='g')
    ax.set_xticks(df['Rok produkcji'])
    ax.set_xticklabels(labels=df['Rok produkcji'], rotation=40)
    plt.show()


def brak_inkaso(df):
    """Relacja pomiędzy rodzajem klienta a niezpłacona składka"""

    df['Data rat'] = pd.to_datetime(df['Data rat'])
    df['Nazwisko'] = df['Nazwisko'].fillna(df['FIRMA'])

    df = df[
            (df['Rozlicz skł. OWCA'].isin(['MAGRO', 'Robert'])) &
            (df['Data rat'] <= (datetime.datetime.today() - timedelta(days=18))) &
            # (df['TUrozlcz?'] == 'do rozl')
            (df['TU Raty'] > 0)
            ]

    all_dates = pd.date_range('2017', '2021.12.01').to_pydatetime()

    print(df.head())
    @plt.FuncFormatter
    def fake_dates(x, pos):
        """Custom formater to turn floats into e.g., 2016-05-08"""
        return dates.num2date(x).strftime('%Y-%m-%d')

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()

    df1 = df[['Data rat', 'TU Raty', 'Rozlicz skł. OWCA']]
    df2 = pd.DataFrame({'Data rat': all_dates})

    """ - Dodać kategorie 'hue' na każdą OFWCA """

    df3 = pd.merge(df2, df1, how='left', on=['Data rat'])
    df3 = df3[['Data rat', 'TU Raty', 'Rozlicz skł. OWCA']]

    x = dates.datestr2num(df3['Data rat'].astype(str))



    """dtype: float64 vs Name: TU Raty, dtype: float64"""
    x = pd.Series(x)
    # df3['TU Raty'] = df3['TU Raty'].astype('Int64', errors='ignore')
    print(df3['TU Raty'])
    print(x)

    print(type(x), type(df3['TU Raty']))




    ax = sns.lmplot(x=x, y='TU Raty', data=df3)#, hue='Rozlicz skł. OWCA')

    # here's the magic:
    ax.xaxis.set_major_formatter(fake_dates)

    # legible labels
    ax.tick_params(labelrotation=45)
    plt.show()


if __name__ == '__main__':
    excel = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    output = 'output.xlsx'

    engine = make_sql(excel)
    sql_df = pd.read_sql('Select * from baza', engine)

    # zapisz_excel(sql_df, output)

    msc = msc()

    # inkaso_agencji(sql_df, msc)
    # przypis_inkaso_agencji(sql_df, msc)
    # inkaso_magro(sql_df, msc)
    # przypis_inkaso_magro(sql_df, msc)
    # displot_przypis(sql_df)
    # displot_rocznik(sql_df)   <--- szkolka
    # rocznik_przypis(sql_df)
    brak_inkaso(sql_df)
