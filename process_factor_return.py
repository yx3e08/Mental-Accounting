import pandas as pd
import numpy as np

path_os_data =   "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/data excel/"
path_os_result = "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/result excel/"

print("Reading Data ...")

list = pd.read_excel(path_os_result + "Ticker_list_dropped.xlsx", skiprows=0)['Ticker']
w_bond = pd.read_excel("w_bond0.xlsx", skiprows=0, index_col='Date')
w_stock = pd.read_excel("w_stock0.xlsx", skiprows=0, index_col='Date')
date = w_bond.index
pn = pd.Panel(items=list, major_axis=date,
              minor_axis=["Return of Bond", "Return of Stock", "Weight of Bond", "Weight of Stock"])


def tot_return(name):
    """
    this function reads and proceeds data
    """
    name = name
    df_1 = pd.read_excel(path_os_data + name + "1.xlsx", skiprows=0)
    df_1.index = df_1['Date']
    df_2 = pd.read_excel(path_os_data + name + "2.xlsx", skiprows=0)
    df_2.index = df_2['Date']
    r_stock_1 = df_1['Equity'] / 100 + 1
    r_stock_2 = df_2['Equity'] / 100 + 1
    r_bond_1 = df_1['Fixed Income'] / 100 + 1
    r_bond_2 = df_2['Fixed Income'] / 100 + 1
    r_stock_1 = r_stock_1.pct_change()
    r_stock_2 = r_stock_2.pct_change()
    r_bond_1 = r_bond_1.pct_change()
    r_bond_2 = r_bond_2.pct_change()
    r_1 = [r_stock_1, r_bond_1]
    r_2 = [r_stock_2, r_bond_2]
    r_1 = pd.DataFrame(r_1).T.dropna()
    r_2 = pd.DataFrame(r_2).T.dropna()
    r = [r_1, r_2]
    r = pd.concat(r)
    return r


"""
def rtn_gmean(rtn):

    this function gets the gmean of return

    rtn = rtn + 1
    rtn_g = gmean(rtn)
    rtn_g = pd.DataFrame(rtn_g, index=['Equity', 'Fixed Income']).T - 1
    return rtn_g
"""


def match_month(df):
    """
    this function converts D to Q
    """
    df = np.log(df + 1)
    df = df.resample('Q').sum()
    return df


print("Calculating ...")
k = 1.0
for i in list:
    vars()[i + 'D'] = tot_return(i)
    vars()[i + '_Daily'] = match_month(tot_return(i))
    vars()[i + '_Daily'].index = date
    pn[i]['Return of Bond'] = vars()[i + '_Daily']['Fixed Income']
    pn[i]['Return of Stock'] = vars()[i + '_Daily']['Equity']
    pn[i]['Weight of Bond'] = w_bond[i]
    pn[i]['Weight of Stock'] = w_stock[i]
    print(i + ":", k / len(list) * 100, "%")
    k += 1

r_bond = pd.DataFrame(index=date, columns=w_bond.columns)
r_stock = pd.DataFrame(index=date, columns=w_stock.columns)
for i in pn.items:
    r_bond[i] = pn[i]["Return of Bond"] / pn[i]['Weight of Bond']
    r_stock[i] = pn[i]["Return of Stock"] / pn[i]['Weight of Stock']

print("Writing ...")
for i in ["r_bond", "r_stock", "pn"]:
    vars()[i] = vars()[i].fillna(0)
    eval(i).to_excel(path_os_result + i + ".xlsx")
    print (eval(i))

print("Done")
