from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import Date
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
#           dtype={'Data wystawienia': Date,
#                  'Przypis': Integer},
#           if_exists='replace')
#
# results = engine.execute("""Select Przypis from baza where Przypis > 9000""")
#
# final = pd.DataFrame(results)  #, columns=df.columns[:152])
# # ## final.to_excel(output, index=False)
# print(final.head())

print()
df = pd.read_sql('Select "Data wystawienia", Przypis, Firma, Nazwisko from baza where Przypis > 6000', engine)
df = df.fillna(df['Nazwisko'])
print(df)

# sns.set(rc={'figure.figsize': (6, 6)});fig, ax = plt.subplots();fig.autofmt_xdate()
# ax = sns.barplot(x='Data wystawienia', y='Przypis', data=df, hue='Firma')
# # ax.set_title('Płeć osób skladajcych zapytania.')
# plt.show()
