import utils
import ppf
import pandapower.networks as pn
import pandas as pd
import numpy as np

cases = 100
iterations = 100

net = pn.mv_oberrhein(scenario='generation')

LoadsP = np.array(pd.read_excel("../support_files/res_bus/p_mw.xlsx", index_col=0))

rndLoadsP, rndLoadsQ = ppf.normal_sampling(LoadsP, cases, iterations)

ppf.plot_dist_bus(rndLoadsP, rndLoadsQ)

net_case = ppf(net, rndLoadsP, rndLoadsQ)

ika_results = net_case.ppf_algorithm(cases, iterations)
