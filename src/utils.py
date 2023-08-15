import pandapower as pp
import pandapower.networks as pn
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import pandapower.control as control
import pandapower.networks as nw
import pandapower.timeseries as timeseries
from pandapower.timeseries.data_sources.frame_data import DFData
import statsmodels as st


# Create models from data
def best_fit_distribution(data, bins=100, ax=None):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    #     DISTRIBUTIONS = [
    #         st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
    #         st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
    #         st.foldcauchy,st.foldnorm,st.frechet_r,st.frechet_l,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
    #         st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
    #         st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #         st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,st.levy,st.levy_l,st.levy_stable,
    #         st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
    #         st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
    #         st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,
    #         st.uniform,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    #     ]

    # DISTRIBUTIONS = [st.beta, st.norm, st.bernoulli,
    #                  st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r, st.gumbel_l,
    #                  st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
    #                 st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone]

    DISTRIBUTIONS = [st.beta, st.norm, st.bernoulli,
                     st.hypsecant, st.invgamma, st.invgauss,
                     st.invweibull, st.ksone]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for distribution in DISTRIBUTIONS:

        # Try to fit the distribution
        try:
            # Ignore warnings from data that can't be fit
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # if axis pass in add to plot
                try:
                    if ax:
                        pd.Series(pdf, x).plot(ax=ax)
                    end
                except Exception:
                    pass

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass

    return best_distribution.name, best_params


def test_data_creation(days=7):
    """
    Based on timeseries pandapower example
    :return:
    """
    # load a pandapower network
    net = pn.mv_oberrhein(scenario='generation')
    # number of time steps
    n_ts = days * 24
    df = pd.DataFrame(np.random.normal(1., 0.1, size=(n_ts, len(net.sgen.index))),
                      index=list(range(n_ts)), columns=net.sgen.index) * net.sgen.p_mw.values
    # create the data source from it
    ds = DFData(df)

    # initialising ConstControl controller to update values of the regenerative generators ("sgen" elements)
    # the element_index specifies which elements to update (here all sgens in the net since net.sgen.index is passed)
    # the controlled variable is "p_mw"
    # the profile_name are the columns in the csv file (here this is also equal to the sgen indices 0-N )
    const_sgen = control.ConstControl(net, element='sgen', element_index=net.sgen.index,
                                      variable='p_mw', data_source=ds, profile_name=net.sgen.index)

    # create a DataFrame with some random time series as an example
    df = pd.DataFrame(np.random.normal(1., 0.1, size=(n_ts, len(net.load.index))),
                      index=list(range(n_ts)), columns=net.load.index) * net.load.p_mw.values
    ds = DFData(df)
    const_load = control.ConstControl(net, element='load', element_index=net.load.index,
                                      variable='p_mw', data_source=ds, profile_name=net.load.index)

    # initialising the outputwriter to save data to excel files in the current folder. You can change this to .json,
    # .csv, or .pickle as well
    ow = timeseries.OutputWriter(net, output_path="../support_files/", output_file_type=".xlsx")
    # adding vm_pu of all buses and line_loading in percent of all lines as outputs to be stored
    ow.log_variable('res_bus', 'vm_pu')
    ow.log_variable('res_line', 'loading_percent')
    ow.log_variable('res_line', 'i_ka')
    ow.log_variable('res_bus', 'p_mw')

    timeseries.run_timeseries(net)


if __name__ == 'main':
    pass
