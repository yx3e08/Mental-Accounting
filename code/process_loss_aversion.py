
"""
Created on Sun Aug 06 20:42:45 2017

@author: Evojoy
"""
# %%
import pandas as pd
import numpy as np
import sympy as sy
from scipy.optimize import minimize
from sympy.utilities.lambdify import lambdify
os_path = "C:/Dropbox/Research Project/mental account/Tickers-Bloomberg/Ticker_moderate_US/"
for i in ["w_stock", "r_stock", "w_bond", "r_bond"]:
    vars()[i] = pd.read_excel(os_path + i + ".xlsx")  # read data and name dataframes on the fly

# get log returns
# r_stock = np.log(1 + r_stock)
# r_bond = np.log(1 + r_bond)

# load data

rfset = list([0.02 / 4]) * w_stock.shape[1]  # suppose risk-free rate in Chinese is 0.02

v = 1.1  # for now v is fixed


def getF(df):  # get the F for a given series
    df = df
    mean = df.mean()
    ddf = df[df >= mean]
    return ddf.count() / df.count()


def dw(Delta, Ft):
    Delta, F = Delta, Ft
    return (Delta / (F)) * (-np.log(F)) ** (Delta - 1) * sy.exp(-(-np.log(F)) ** Delta)


df_equity_F = getF(r_stock)
df_bond_F = getF(r_bond)

# get average return for the entire sample
df_equity_benchmark, df_bond_benchmark = r_stock.mean(), r_bond.mean()

# initialize estimating parameters
la, phi, delta = sy.symbols('la, phi, delta')
equity_foc_arry = sy.zeros(r_stock.shape[0], r_stock.shape[1])
bond_foc_arry = sy.zeros(r_stock.shape[0], r_stock.shape[1])
or_foc_arry = sy.zeros(r_stock.shape[0], r_stock.shape[1])


def make_U(LA, Phi, Delta, alpha, r, Mu, F, rf):
    phi = Phi
    A = LA
    mu, a, Delta = Mu, alpha, Delta
    if r >= Mu:
        up = (((r - Mu)) ** v) * dw(Delta, F)
        down = 0
    else:
        up = 0
        down = ((-(r - Mu)) ** v) * dw(Delta, 1 - F)
    # print down,up
    # estimated_LA = ((a ** (1-v))*(mu-rf))/(phi*v* down)+ 1
    U = (r - rf) - phi * v * a ** (v - 1) * 1 * (A * down - up)
    return U


def Gmm_Minimizor(fun, par = [2.25, 1.5, .741] ):
    func = fun
    my_func = lambdify((la, phi, delta), func)

    def my_func_v(x):
        return my_func(*tuple(x))

    # results = minimize(my_func_v,[2.25, 1.5, 1], method= 'BFGS', tol= 0.000001, options={'gtol': 1e-8})
    # bnds = ((0, 4), (0, 3), (0,1))
    #results = minimize(my_func_v, par, method='L-BFGS-B')

    results = minimize(my_func_v, par, method='Nelder-Mead')
    # results = minimize(my_func_v, [LA, 1., 1], method='Nelder-Mead')

    return results.x


def hassian(fun, optimal):
    """
    :param fun: the fun you optimize
    :param optimal: numpy series
    :return: numpy series of estimates' standard errors
    """
    x, y, z = optimal[0], optimal[1], optimal[2]
    d2_la = sy.diff(fun, la, 2)
    d2_phi = sy.diff(fun, phi, 2)
    d2_delta = sy.diff(fun, delta, 2)
    return np.array([float(d2_la.subs({la: x, phi: y, delta: z})),
                     float(d2_phi.subs({la: x, phi: y, delta: z})),
                     float(d2_delta.subs({la: x, phi: y, delta: z}))]) ** 0.5


# Make GMM vectors
for i in range(r_stock.shape[1]):
    Mu = df_equity_benchmark[i]
    rf = rfset[i]
    F = df_equity_F[i]
    for j in range(r_stock.shape[0]):
        a = w_stock.iloc[j, i]
        r = r_stock.iloc[j, i]
        equity_foc_arry[j, i] = make_U(la, phi, delta, a, r, Mu, F, rf)
foc_equity = np.mean(equity_foc_arry, axis=0)

for i in range(r_stock.shape[1]):
    Mu = df_bond_benchmark[i]
    rf = rfset[i]
    F = df_bond_F[i]
    for j in range(r_stock.shape[0]):
        a = w_bond.iloc[j, i]
        r = r_bond.iloc[j, i]
        bond_foc_arry[j, i] = make_U(la, phi, delta, a, r, Mu, F, rf)
foc_bond = np.mean(bond_foc_arry, axis=0)

estimate_stock, estimate_bond = pd.DataFrame(), pd.DataFrame()
for i in range(len(foc_equity)):
    add_stock = pd.DataFrame(Gmm_Minimizor((foc_equity[i]) ** 2))
    add_bond = pd.DataFrame(Gmm_Minimizor((foc_bond[i]) ** 2)) #use different guess values for bonds
    estimate_stock = pd.concat([estimate_stock, add_stock.T])
    estimate_bond = pd.concat([estimate_bond, add_bond.T])

estimate_stock.index = w_stock.columns
estimate_bond.index = w_stock.columns
estimate_stock.columns = ["LA_s", "Phi_s", "Delta_s"]
estimate_bond.columns = ["LA_b", "Phi_b", "Delta_b"]

df = pd.concat((estimate_stock, estimate_bond), axis=1)
dff = df[(df["LA_s"] > 0) & (df["LA_b"] > 0)]

print(df)
print(dff)
dff.to_excel(os_path + "Loss Aversion.xlsx")
print("Done")
# value = foc_equity[0].subs({la: 2, phi: 1, delta: 1})