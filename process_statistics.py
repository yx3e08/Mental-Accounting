import pandas as pd
path_os_data =   "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/data excel/"
path_os_result = "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/result excel/"

list = pd.read_excel(path_os_result + "Ticker_list_dropped.xlsx", skiprows=0)['Ticker']
w_stat = pd.DataFrame(index=list)
r_stat = pd.DataFrame(index=list)
for i in ["r_bond", "r_stock"]:
    vars()[i] = pd.read_excel(path_os_result + i + ".xlsx")  # annual return stats
    r_stat[i + ".mean"] = vars()[i].mean() * 4
    r_stat[i + ".median"] = vars()[i].median() * 4
    r_stat[i + ".max"] = vars()[i].max() * 4
    r_stat[i + ".min"] = vars()[i].min() * 4
    r_stat[i + ".samplesize"] = len(vars()[i])
    r_stat[i + ".std"] = vars()[i].std() * 2

for i in ["w_bond", "w_stock"]:
    vars()[i] = pd.read_excel(path_os_result + i + ".xlsx")
    w_stat[i + ".mean"] = vars()[i].mean()
    w_stat[i + ".median"] = vars()[i].median()
    w_stat[i + ".max"] = vars()[i].max()
    w_stat[i + ".min"] = vars()[i].min()
    w_stat[i + ".samplesize"] = len(vars()[i])
    w_stat[i + ".std"] = vars()[i].std()



r_stat.to_excel("r_stat.xlsx")
w_stat.to_excel("w_stat.xlsx")
print("Done")