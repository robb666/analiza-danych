from sqlalchemy import create_engine
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy import func
from sqlalchemy import DateTime
import sqlite3
import pandas as pd

Base = declarative_base()

file = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite:///baza_sql.db', echo=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# df = pd.read_excel(file, sheet_name='BAZA 2014', header=1)
# new_header = df.iloc[0]
# df = df[2:]
# df.columns = new_header
#
# df.to_sql('baza', engine,
#           dtype={'Data wystawienia': DateTime(),
#                  'Przypis': Integer()},
#           if_exists='replace')
#
# results = engine.execute("""Select Przypis from baza where Przypis > 9000""")
# #
# final = pd.DataFrame(results)  #, columns=df.columns[:152])
# # ## final.to_excel(output, index=False)
# print(final.head())



class User(Base):
    __tablename__ = 'baza'

    key = Column('index', Integer, primary_key=True)
    data = Column('Data wystawienia', DateTime())
    przypis = Column('Przypis', Integer())

    def __init__(self, key, data, przypis):
        self.key = key
        self.data = data
        self.przypis = przypis


# Base.metadata.create_all(bind=engine)
session = sessionmaker(bind=engine)()
# results = session.query(engine).all()
query = session.query(User).filter(User.key, User.data > '2021-11-01', User.przypis).all()

# print(f'Index     Data                Przypis')
# for user in query:
#     print(f'{user.key} -> {user.data} {user.przypis}')


# df = pd.DataFrame([(d.key, d.data, d.przypis) for d in query],
#                   columns=['Index', 'Data', 'Przypis']).round()


df = pd.read_sql('baza', engine)
print(df)
