import pandas as pd
import cx_Oracle
import sqlalchemy

from sqlalchemy.exc import SQLAlchemyError

def getoracledataset(tableorviewname):
    schema = input("Enter connection schema:")
    schemapwd = input("Enter connection schema password:")
    server = input("Enter full server name:")
    serversid = input("Enter Server SID:")
    tablename = tableorviewname

    oracle_connection_string = 'oracle+cx_oracle://' + schema + ':' + schemapwd + '@' + server + ':1521/' + serversid
    engine = sqlalchemy.create_engine(oracle_connection_string)

    data = pd.read_sql("SELECT * FROM " + tablename, engine)

    #For test purposes load to csv file and print
    data.to_csv('test_' + tablename + '.csv')
    print(data.head(25))


    #TODO - Check do I need to specifically need to close connection here
    engine.dispose()

    return data
