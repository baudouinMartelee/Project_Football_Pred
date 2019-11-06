import sqlite3
import pandas as pd

dat = sqlite3.connect('data/database/database.sqlite')

table = ["Country", "League", "X_Train", "Player", "Player_Attributes",
         "Team", "Team_Attributes"]

csv = {}

for name in table:
    query = dat.execute("SELECT * From " + name)
    cols = [column[0] for column in query.description]
    results = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    #csv[name] = results
    results.to_csv(r''+name+'.csv')
    #print(name + " shape ==> " + str(results.shape))
