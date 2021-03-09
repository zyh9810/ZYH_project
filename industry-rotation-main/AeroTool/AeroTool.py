from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import math

class AeroTool():
    def __init__(self):
        pass

    # this func plot the industry nav and the ideal nav using given pressures.
    # @bench indicate the industry itself, @ideal indicate the prescribed strategy.
    # in the bottom figure, the solid line indicate buy pressure, the dotted line
    # indicate sell pressure, and both are large when value are postive.
    @staticmethod
    def Xray_standalone(ticker,conf,startDay=None, endDay=None):
        q = 252
        perf = pd.DataFrame(columns=['bench', 'ideal'], index=['ret', 'risk', 'sr'])

        ass = pd.read_csv("{}.csv".format(ticker), header=0,dtype={'tradingDay': str})

        holding = ass['ideal'].shift(periods=1).fillna(value=0)

        ass['nav-bench'] = ass['logR'].cumsum().apply(lambda x: np.exp(x))
        #@lr is a pd.Series of log returns.
        def _performance(lrets,q,rf=0.00):
            lrets = lrets.fillna(value=0)
            excess = np.exp(lrets.mean()*q)-rf-1
            daily_gain = lrets.apply(lambda x: np.exp(x)-1)
            risk = daily_gain.std()*np.sqrt(q)
            sr = excess/risk
            return excess,risk,sr

        perf.loc[:, 'bench'] = _performance(ass['logR'], q)

        daily_gain = ass['logR'].apply(lambda x: np.exp(x) - 1) * holding
        ass['nav-ideal'] = (daily_gain + 1).cumprod()
        perf.loc[:, 'ideal'] = _performance(daily_gain.apply(lambda x: np.log(1 + x)), q)

        if startDay!=None:
            ass = ass[ass['tradingDay'] > startDay]
        if endDay!=None:
            ass = ass[ass['tradingDay'] < endDay]

        # ass = ass.set_index('tradingDay')
        ass['tradingDay_index'] = ass['tradingDay'].apply(lambda x: datetime.strptime(x, "%Y%m%d").date())
        ass = ass.set_index('tradingDay_index')

        fig = plt.figure()
        gs = GridSpec(3, 3)
        ax1 = plt.subplot(gs[:2, :])
        ax2 = plt.subplot(gs[2:3, :], sharex=ax1)

        lns1 = ax1.plot(ass['nav-bench'], color='black',
                        label='bench(left):ret={:.2f},risk={:.2f},sr={:.2f}'.format(perf.iloc[0, 0], perf.iloc[1, 0],
                                                                              perf.iloc[2, 0]))
        ax1_ = ax1.twinx()
        lns2 = ax1_.plot(ass['nav-ideal'], color='red',
                         label='ideal(right):ret={:.2f},risk={:.2f},sr={:.2f}'.format(perf.iloc[0, 1], perf.iloc[1, 1],
                                                                               perf.iloc[2, 1]))
        ax1.grid()

        ax2.plot(ass['buyP'], linestyle='-', color='black')
        ax2.plot(ass['sellP'], linestyle=':', color='black')
        ax2_ = ax2.twinx()
        ax2_.plot(ass['ideal'], color='red')
        ax2.grid()

        # added these three lines
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=2)

        ax1.set_title('Xray[{}]: conf=[ratio={},thres={},granu={}], avgH={:.2f}'.
                      format(ticker, conf['gradient_ratio'],conf['port_thres'], conf['granularity'], ass['ideal'].mean()))

        # ax2.legend()
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax2.xaxis.set_major_locator(mticker.MultipleLocator(150))
        ax2.set_xticklabels(labels=ass.index, rotation=90)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))

        plt.savefig("Xray-{}.png".format(ticker))
        plt.show()
        ass.iloc[:,:-2].to_csv("{}.csv".format(ticker), index=False)

    # @long_weight is a pd.Series about buy pressure
    # @short_weight is a pd.Series about sell pressure
    # this func return the combined holding.
    @staticmethod
    def combineWeight(long_weight, short_weight, conf=None):
        if conf==None:
            conf = {"gradient_ratio": 1.2, "port_thres": 10,"granularity":0.05}

        gradient_long = float(conf['granularity'])
        gradient_short = float(conf['gradient_ratio']) * gradient_long
        port_thres = float(conf['port_thres'])

        combined_weight = pd.DataFrame(np.zeros([len(long_weight),2]), index=long_weight.index, columns=['Weight',"Delta"])
        last_port = 0.0
        for t in range(len(long_weight)):
            target_date = long_weight.index[t]
            delta = -gradient_long * long_weight.loc[target_date] + gradient_short * short_weight.loc[target_date]
            current_port = last_port + delta
            if current_port > port_thres:
                current_port = port_thres
            if current_port < 0:
                current_port = 0
            combined_weight.loc[target_date, "Weight"] = current_port
            combined_weight.loc[target_date, "Delta"] = delta

            if math.isnan(current_port):
                last_port = 0
            else:
                last_port = current_port

        combined_weight = combined_weight / port_thres

        return combined_weight


