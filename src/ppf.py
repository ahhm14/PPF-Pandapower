import pandapower as pp
import pandapower.networks as pn
import numpy as np
import pandas as pd
import random
import time
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import timeit
import time
import tabulate
from math import sin, acos


def uniform_sampling(LoadsP, iterations):
    return uniLoadsP, uniLoadsQ


def normal_sampling(LoadsP, cases, iterations):
    """
    This function performs a normal sampling from the measurements for P and Q
    prints the time taken in total
    :return: LoadsP and LoadsQ [bus, cases, iterations]
    """
    num_loads = LoadsP.shape[0]
    start = time.time()
    rndLoadsP = np.zeros((num_loads, cases, iterations))
    rndLoadsQ = np.zeros((num_loads, cases, iterations))

    sigmaP = np.array([np.std(LoadsP[i]) for i in range(LoadsP.shape[0])])
    # sigmaQ = np.array([np.std(LoadsQ[i]) for i in range(LoadsQ.shape[0])])
    #  -----
    print("---- Normal Sampling ----")
    for load in range(num_loads):
        # print(bus)
        for case in range(cases):
            for it in range(iterations):
                # xrvsP = sts.norm.rvs(LoadsP[bus,i], sigmaP[bus])
                xrvsP = np.random.normal(LoadsP[load, case],
                                         LoadsP[load, case] * 0.1)  # The mu,sigma of this are the entries for NN
                rndLoadsP[load, case, it] = xrvsP
                xrvsQ = (xrvsP * 0.95) * sin(acos(0.95))  # decision on how to use Qkvar
                rndLoadsQ[load, case, it] = xrvsQ

    # -------------- END OF SAMPLING ----------------------
    end = time.time()

    time_t = end - start

    if iterations > 999:
        print("Total Execution time ", time_t / 60, "minutes")
    else:
        print("Total Execution time ", time_t, "seconds")

    return rndLoadsP, rndLoadsQ


# add the cong_vector
def plot_dist_bus(rndLoadsP, rndLoadsQ):
    """
    Plots for the probability density function of the normal arrays
    :return: PDF plots
    """
    num_loads = rndLoadsP.shape[0]
    for load in [3, len(num_loads) - 1]:
        sns.histplot(rndLoadsP[load, 0, :], label='Active Power', kde=True)
        sns.histplot(rndLoadsQ[load, 0, :], label='Reactive Power', kde=True, color='orange')
        plt.legend()
        plt.title('Load {}'.format(load))
        plt.xlabel('P(MW) and Q(MVAR)')
        # plt.xlabel('P(MW)')
        #plt.savefig(r"..\files\out\figures\Load_{}.png".format(load))
        plt.show()
        break


# Class to perform probabilistic power flow
class ppf:

    def __init__(self, net, rndLoadsP, rndLoadsQ):
        self.net = net
        self.loadsP = rndLoadsP
        self.loadsQ = rndLoadsQ

    def update_load_vals(self, case, index):
        """
        Function to update the values of the loads to the net object
        :param case: number of cases
        :param index: index of iterations
        :return: a new net object with the changed loads
        """
        # always reset the grid and values
        net2 = self.net
        new_vals_p = [self.loadsP[bus, case, index] for bus in range(self.loadsP.shape[0])]
        # new_vals_q = [self.LoadsP[bus, case, index] for bus in range(self.LoadsQ.shape[0])]
        # setting LoadsQ = 0 -> PF = 1
        new_vals_q = [0] * self.loadsP.shape[0]

        p_vals = pd.Series(new_vals_p, name='p_mw')
        q_vals = pd.Series(new_vals_q, name='q_mvar')

        # Changing the values of the columns
        net2.load.update(p_vals)
        net2.load.update(q_vals)

        return net2

    def ppf_algorithm(self, cases=10, iterations=10):
        """
        This function performs the powerflows set for the MonteCarlo. Prints the time taken for the calculation.
        :param net:
        :param vbus: if voltages results are wanted
        :param cases:
        :param iterations:
        :return: i_ka results [case, i, n_lines], v_bus [case, i, n_bus]
        """
        ika_results = np.zeros((cases, iterations, len(self.net.line)))
        # v_bus = np.zeros((cases, iterations, len(self.net.bus)))
        times = list()
        print("--- PPF Algorithm ---")
        print('Executing {} power flows'.format(cases * iterations))
        start = time.time()
        for case in range(cases):
            if case % 50 == 0:
                print("Executing Case {} out of {}".format(case, cases))
                times.append(time.time() - start)
            for i in range(iterations):
                net2 = self.update_load_vals(case, i)
                # print(net.load)
                pp.runpp(net2)
                # print(net.res_line)
                ika_results[case, i, :] = net2.res_line['i_ka'].sort_index()
                # v_bus[case, i, :] = net2.res_bus['vm_pu'].sort_index()
                # break
        end = time.time()
        time_t = end - start
        print("Total Execution time ", time_t / 60, "minutes")
        return ika_results


def plot_results(net, ika_mean, ika_std, cong_lines, cong_vector, limit_set):
    for line in cong_lines[1]:
        print('---------------')
        print('Line {}'.format(line))
        for x in cong_vector:
            value = np.random.normal(loc=ika_mean[x, line], scale=ika_std[x, line], size=2000)
            sns.histplot(value, label='PPF', kde=True)
            plt.vlines(net.line.max_i_ka[line], ymin=0, ymax=200,
                       linestyles='dashed', label='Limit', colors='black')
            plt.vlines(net.line.max_i_ka[line] * limit_set, ymin=0, ymax=200,
                       linestyles='dashed', label='{} % Limit'.format(limit_set), colors='orange')
            plt.legend()
            plt.xlabel('kA')
            plt.ylabel('freq')
            plt.title('Line {} - Case {}'.format(line, x))
            # plt.savefig(r"../files/out/figures/Line {} - Case {}".format(line, x))
            plt.show()


def plot_results3(net, Ika_mean, Ika_std, cong_line, cong_time, limit_set):
    for line in cong_lines[:2]:
        print('---------------')
        print('Line {}'.format(line))
        value = np.random.normal(loc=Ika_mean[cong_time, line], scale=Ika_std[cong_time, line], size=2000)
        sns.histplot(value, label='PPF', kde=True)
        plt.vlines(net.line.max_i_ka[line], ymin=0, ymax=200,
                   linestyles='dashed', label='Limit', colors='black')
        plt.vlines(net.line.max_i_ka[line] * limit_set, ymin=0, ymax=200,
                   linestyles='dashed', label="limit set", colors='orange')
        plt.xlim(0, net.line.max_i_ka[line] * 1.10)
        plt.legend()
        plt.xlabel('kA')
        plt.ylabel('freq')
        plt.title('Line {} - Case {}'.format(line, cong_time))
        # plt.savefig(r"../files/out/figures/Line {} - Case {}".format(line, x))
        plt.show()


def plot_line_dist(net, cong_lines, ika_results, cong_times, load_factor):
    for i, line in enumerate(cong_lines):
        print('---------------')
        print('Line {}'.format(line))
        sns.histplot(ika_results[list(cong_times)[i], :, line], label='PPF', kde=True)
        plt.vlines(net.line.max_i_ka[line], ymin=0, ymax=200,
                   linestyles='dashed', label='Limit', colors='black')
        plt.vlines(net.line.max_i_ka[line] * load_factor, ymin=0, ymax=200,
                   linestyles='dashed', label='{} % Limit'.format(load_factor * 100), colors='orange')
        plt.legend()
        plt.xlabel('kA')
        plt.ylabel('freq')
        plt.title('Line {} - Case {}'.format(line, cong_times.iloc[i]))
        # plt.savefig(r"files/out/" + str(test_case) + "/figures/Line {} - Case {}".format(line, cong_times[i]))
        plt.show()