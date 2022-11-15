# https://data.nasdaq.com/tools/api
import nasdaqdatalink

nasdaqdatalink.ApiConfig.api_key = "JEg17RDmuJqLE6KY-WvN"
mydata = nasdaqdatalink.get_table('ZACKS/FC', ticker='AAPL')

print(mydata)
