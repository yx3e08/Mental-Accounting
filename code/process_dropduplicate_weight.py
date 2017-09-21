import pandas as pd
#hello
path_os_data = "C:/Dropbox/Research Project/mental account\Tickers-Bloomberg/Ticker_moderate_US/data excel/"
path_os_result = "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/result excel/"
# read data
read = pd.read_excel(path_os_data + "Ticker_list.xlsx")
list = read['Ticker'].str[0: 5]
date = read['Date'][0:25]
s = pd.DataFrame(index=date, columns=list).drop('Security')
b = pd.DataFrame(index=date, columns=list).drop('Security')

k = 1.0
for i in list:
    df = pd.read_excel(path_os_data + i + ".xlsx")
    equity_column = df['Unnamed: 1']
    equity_index = equity_column.loc[equity_column == 'Equity'].index
    bond_index = equity_column.loc[equity_column == 'Fixed Income'].index

    if 'Funds' in equity_column:
        funds_index = equity_column.loc[equity_column == 'Funds'].index
    else:
        funds_index = 10000000

    dff = df.loc[df.index.values > equity_index.values]
    dfff = dff.loc[dff.index.values < funds_index]

    dfff = dfff.drop(['Unnamed: 1', 'Unnamed: 2'], axis=1)

    if len(dfff.columns) > 26:
        dfff = dfff.drop('Unnamed: 4', axis=1)
    else:
        pass

    dfff = dfff.drop(dfff.columns[0], axis=1)
    name = dfff['Unnamed: 3'].dropna()
    dfff = dfff.loc[name.index]
    dfff.columns = date
    stock = dfff.loc[dfff.index.values < bond_index.values]
    bond = dfff.loc[dfff.index.values > bond_index.values]
    stock = stock.sum().drop('Security')
    bond = bond.sum().drop('Security')
    s[i] = stock
    b[i] = bond
    print(i + ":", k / len(list) * 100, "%")
    k += 1

s = s.T.drop_duplicates().fillna(0) / 100
b = b.T.drop_duplicates().fillna(0) / 100
s.T.to_excel(path_os_result + "w_stock.xlsx")
b.T.to_excel(path_os_result + "w_bond.xlsx")
list_d = pd.DataFrame(index=s.index)
list_d.to_excel(path_os_result + "Ticker_list_dropped.xlsx")
print ("Done")