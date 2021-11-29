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

file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite:///baza_sql.db', echo=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# df = pd.read_excel(file, sheet_name='BAZA 2014', header=1)
# new_header = df.iloc[0]
# df.columns = new_header
# df = df[2:]
# df.reset_index()
#
# df.to_sql('baza', engine,
#           dtype={
#                  'Data wystawienia': Date,
#                  'Przypis': Integer,
#                  'TU Inkaso': Integer,
#                  },
#           if_exists='replace')
#
# # results = engine.execute("""Select * from baza""")
# ## final.to_excel(output, index=False)
#
# # print(final)



# print()
df = pd.read_sql('Select * from baza', engine)
# # df2 = df['FIRMA'].fillna(df['Nazwisko'])

df = df.rename(columns={'TU Inkaso': 'Inkaso w PLN --> przychód'})
df_msc = pd.Series(df['Miesiąc przypisu'].replace({'_': ''}, regex=True))#.str.slice(stop=2)

df.insert(3, 'Strzałka czasu', df_msc)
df = df.sort_values(by=['Strzałka czasu'])
dff = df.groupby(['Strzałka czasu']).sum()
rok_msc = df['Strzałka czasu'].unique()
print(rok_msc)

sns.set(rc={'figure.figsize': (7, 7)});fig, ax = plt.subplots();fig.autofmt_xdate()
ax = sns.lineplot(x='Strzałka czasu', y='Inkaso w PLN --> przychód', data=dff, ci=68)
# ax = sns.lineplot(x='Strzałka czasu', y='Przypis', data=dff)
msc = ['styczeń', 'luty', 'marzec', 'kwiecień', 'maj', 'czerwiec', 'lipiec',
       'sierpień', 'wrzesień', 'październik', 'listopad', 'grudzień']
rok = [rok[:2] for rok in rok_msc if rok is not None]
ax.set_xticklabels(labels=[f'\'{rok} {msc}' for rok, msc in zip(rok, cycle(msc))], rotation=40)
ax.set_title('INKASO')
plt.show()
