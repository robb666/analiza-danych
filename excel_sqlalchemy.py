from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import DateTime
from datetime import datetime
import sqlite3
import pandas as pd


input = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite:///baza_sql.db', echo=False)


pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# df = pd.read_excel(input, sheet_name='BAZA 2014', header=1)
# new_header = df.iloc[0]
# df = df[1:]
# df.columns = new_header
# print(df.head())
# df.to_sql('baza', engine,
#           dtype={'Data wystawienia': DateTime(),
#                  'Przypis': Integer()})  # , if_exists='replace', index_label=False)

results = engine.execute('Select * from baza where (Imie="Adam" or Imie="ADAM")'
                         'and Przypis <= 100 '
                         'and "Data wystawienia">="2017-02-25" ')

final = pd.DataFrame(results)#, columns=df.columns[:152])
# final.to_excel(output, index=False)

print(final)
