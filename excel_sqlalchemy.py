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

input = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite:///baza_sql.db', echo=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_excel(input, sheet_name='BAZA 2014', header=1)
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
print(df.head())
df.to_sql('baza', engine,
          dtype={'id': Column('id', Integer, primary_key=True),
                 'Data wystawienia': DateTime(),
                 'Przypis': Integer()})  # , if_exists='replace', index_label=False)

results = engine.execute(
    """
    Select * from baza where (Imie="Adam" or Imie="ADAM")
    and Przypis <= 100
    and "Data wystawienia">="2020-07-20 00:00:00.000000"
    """)


# class User(Base):
#     __tablename__ = 'user'
#
#     key = Column('id', Integer, primary_key=True)
#     data = Column('Data wystawienia', DateTime())
#     przypis = Column('Przypis', Integer())
#     # stardust = Column('stardust', Integer, unique=True)
#
#     def __init__(self, data, przypis):
#         self.username = data
#         self.password = przypis
#
#
# Base.metadata.create_all(bind=engine)
# session = sessionmaker(bind=engine)()
# # results = session.query(engine).all()
# users = session.query(User).all()
# for user in users:
#     print(user)

final = pd.DataFrame(results)  #, columns=df.columns[:152])
# final.to_excel(output, index=False)

print(final)
