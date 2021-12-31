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
    pd.options.mode.chained_assignment = None

    df_income = db[
                      # (db['Rok rozlicz nr PERITUS'].str.contains('_21', na=False))
                      (db['Rok rozlicz nr PERITUS'].str.contains('|'.join(['_20', '_21']), na=False))
    ]

    print(df_income['TU Inkaso'].sum())

    sns.set(rc={'figure.figsize': (29, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()

    df_income['TU nr rozlicz prowizji'] = df_income['TU nr rozlicz prowizji'].str.extract('\w+(\d{4})')
    df_income.dropna(subset=['TU nr rozlicz prowizji'], inplace=True)
    df_income['TU nr rozlicz prowizji'] = pd.to_datetime(df_income['TU nr rozlicz prowizji'], format='%y%m')
    df_income = df_income.groupby(['TU nr rozlicz prowizji']).sum().reset_index()
    df_income = df_income[df_income['TU nr rozlicz prowizji'] < pd.to_datetime('2021-12')]

    df_income['TU Inkaso'] = df_income['TU Inkaso'].fillna(0).astype(int)
    x = df_income['TU nr rozlicz prowizji']

    ax.xaxis.update_units(x)

    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='TU Inkaso',
                     data=df_income,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4, 'color': 'green'},
                     line_kws={'lw': 1, 'color': 'green'},
                     label="Przychody")

    df_bank.Kwota = df_bank.Kwota.replace({',': '.'}, regex=True)
    df_bank.Kwota = df_bank.Kwota.astype(float)

    df_bank['Data nowa'] = df_bank['Data księgowania'].apply(lambda x: x[3:])

    df_costs = df_bank[['Data nowa', 'Nadawca', 'Kwota']]
    df_costs['Nadawca'] = df_costs['Nadawca'].fillna('bez tyt.')

    customers_premium = df_costs.iloc[[
                                          140, 160, 295, 327, 353,   # 2020
                                          596, 599, 610, 620, 622,   # 2021
                                          651, 709, 777, 789, 824,   # 2021
                                          836, 867, 895, 905, 960,   # 2021
                                          966, 985, 1101, 1131, 1150,# 2021
                                          1159]]                     # 2021

    customers_premium.Kwota = customers_premium.Kwota * -1

    mg = df_costs[df_costs['Nadawca'].str.contains('MAGRO MACIEJ')]
    mg.Kwota = mg.Kwota * -1

    df_costs = df_costs[df_costs['Kwota'] < 0]
    df_costs.Kwota = df_costs.Kwota * -1

    df_costs = pd.concat([df_costs, mg, customers_premium], axis=1, ignore_index=False)
    df_costs = df_costs.groupby(df_costs.columns, axis=1).sum()

    df_costs = df_costs.groupby(['Data nowa']).sum().reset_index()
    df_costs.Kwota = df_costs.Kwota.astype(int)

    df_costs['Data nowa'] = pd.to_datetime(df_costs['Data nowa'])
    df_costs = df_costs.sort_values('Data nowa')
    df_costs = df_costs[(df_costs['Data nowa'] > pd.to_datetime('2020-04')) &
                        (df_costs['Data nowa'] < pd.to_datetime('2021-12'))]

    x = df_costs['Data nowa']
    ax.xaxis.update_units(x)

    ax = sns.regplot(x=ax.xaxis.convert_units(x),
                     y='Kwota',
                     data=df_costs,
                     order=2,
                     scatter_kws={'s': 10, 'alpha': 0.4, 'color': 'red'},
                     line_kws={'lw': 1, 'color': 'r'},
                     label='Koszty')

    df_income = df_income[(df_income['TU nr rozlicz prowizji'] > pd.to_datetime('2020-12')) &
                          (df_income['TU nr rozlicz prowizji'] < pd.to_datetime('2021-12'))]

    df_costs = df_costs[(df_costs['Data nowa'] > pd.to_datetime('2020-12')) &
                        (df_costs['Data nowa'] < pd.to_datetime('2021-12'))]

    income = df_income["TU Inkaso"].sum() - df_costs.Kwota.sum()

    print(f"""
    \nSuma Inkasa z Bazy: {df_income['TU Inkaso'].sum()} zł
    \nSuma Kosztów z konta: {df_costs.Kwota.sum()} zł
    \nInkaso minus Koszty: {income} zł""")

    ax.legend()
    ax.legend()
    ax.set_title('Pównanie przychodów z Bazy _20, _21 i kosztów od 05.2020 (tylko Spółka bez jdg).\n\n'
                 f'Dochód w 2021 bez grudnia = {income} zł')
    plt.show()


if __name__ == '__main__':
    # file = "/run/user/1000/gvfs/smb-share:server=192.168.1.12,share=e/Agent baza/2014 BAZA MAGRO.xlsx"
    db_file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
    bank_statement = '/home/robb/Desktop/historia_2021-12-30_20109027050000000133736204.csv'

    engine = make_sql(db_file)

    sql_df = pd.read_sql('Select * from baza', engine)

    bank = read_bank(bank_statement)

    print(plot(sql_df, bank))
