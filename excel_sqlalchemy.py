from sqlalchemy import create_engine
import sqlite3
import pandas as pd


input = '/home/robb/Desktop/2014 BAZA MAGRO.xlsx'
output = 'output.xlsx'

engine = create_engine('sqlite://', echo=False)


pd.set_option('display.max_rows', 40)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

df = pd.read_excel(input, sheet_name='BAZA 2014', header=1)
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
print(df)

df.to_sql('baza', engine)#, if_exists='replace', index_label=False)
results = engine.execute('Select * from baza where Imie="Robert" and Przypis >= 1000')
final = pd.DataFrame(results)#, columns=df.columns[:152])
final.to_excel(output, index=False)

print(final)