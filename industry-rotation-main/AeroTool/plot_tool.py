'''
作图工具包
'''

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from datetime import datetime
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd

def plot_nav(ass,ticker,startDay=None, endDay=None,conf=None):
    '''
    此函数为净值曲线的画图函数
    :param ass:  是作图函数的数据，格式为dataframe，需要包括：
    tradingDay:交易日期
    predict:预测仓位
    ideal:理想仓位
    logR：对数收益率

    :param ticker:申万二级行业代码
    :param startDay:开始日期
    :param endDay:结束日期
    :param conf:配置项
    :return:
    '''
    q = 252
    if conf is None:
        conf = {"gradient_ratio": 1.2, "port_thres": 10, "granularity": 0.1}

    perf = pd.DataFrame(columns=['bench', 'predict'], index=['ret', 'risk', 'sr'])
    # ass = pd.read_csv("{}.csv".format(ticker), header=0, dtype={'tradingDay': str})

    holding = ass['predict'].shift(periods=1).fillna(value=0)

    ass['nav-bench'] = ass['logR'].cumsum().apply(lambda x: np.exp(x))

    # @lr is a pd.Series of log returns.
    def _performance(lrets, q, rf=0.00):
        lrets = lrets.fillna(value=0)
        excess = np.exp(lrets.mean() * q) - rf - 1
        daily_gain = lrets.apply(lambda x: np.exp(x) - 1)
        risk = daily_gain.std() * np.sqrt(q)
        sr = excess / risk
        return excess, risk, sr

    perf.loc[:, 'bench'] = _performance(ass['logR'], q)
    daily_gain = ass['logR'].apply(lambda x: np.exp(x) - 1) * holding
    daily_gain_bench=ass['logR'].apply(lambda x: np.exp(x) - 1)
    ass['nav-predict'] = (daily_gain + 1).cumprod()
    perf.loc[:, 'predict'] = _performance(daily_gain.apply(lambda x: np.log(1 + x)), q)
    ass["nav-alpha"]= np.cumprod(daily_gain - daily_gain_bench + 1)
    alpha_ret=np.exp((daily_gain-daily_gain_bench).mean()*q)-1
    IR=(perf.iloc[0,1]-perf.iloc[0,0])/((daily_gain-daily_gain_bench).std()*np.sqrt(q))

    if startDay != None:
        ass = ass[ass['tradingDay'] > startDay]
    if endDay != None:
        ass = ass[ass['tradingDay'] < endDay]

    # ass = ass.set_index('tradingDay')
    ass['tradingDay_index'] = ass['tradingDay'].apply(lambda x: datetime.strptime(x, "%Y%m%d").date())
    ass = ass.set_index('tradingDay_index')

    fig = plt.figure(figsize=(9,7))
    gs = GridSpec(3, 3)
    ax1 = plt.subplot(gs[:2, :])
    ax2 = plt.subplot(gs[2:3, :], sharex=ax1)

    lns1 = ax1.plot(ass['nav-bench'], color='black',
                    label='bench:ret={:.2f},risk={:.2f},sr={:.2f}'.format(perf.iloc[0, 0], perf.iloc[1, 0],
                                                                                perf.iloc[2, 0]))
    # ax1_ = ax1.twinx()
    lns2 = ax1.plot(ass['nav-predict'], color='red',
                     label='predict:ret={:.2f},risk={:.2f},sr={:.2f}'.format(perf.iloc[0, 1], perf.iloc[1, 1],
                                                                                  perf.iloc[2, 1]))
    lns3 = ax1.plot(ass['nav-alpha'],color="green",label='alpha:ret={:.2f},IR={:.2f}'.format(alpha_ret,IR))
    ax1.grid()
    #
    # ax2.plot(ass['buyP'], linestyle='-', color='black')
    ax2.plot(ass['ideal'], linestyle='-', color='black',label='ideal:mean={:.3f}'.format(ass['ideal'].mean()))
    ax2.plot(ass['predict'], color='red', label='predict:mean={:.3f}'.format(ass['predict'].mean()))
    ax2.grid()

    # added these three lines
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=2)

    ax2.legend(loc=2)

    ax1.set_title('Xray[{}]: conf=[ratio={},thres={},granu={}], avgH={:.2f}'.
                  format(ticker, conf['gradient_ratio'], conf['port_thres'], conf['granularity'], ass['predict'].mean()))

    # ax2.legend()
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(150))
    ax2.set_xticklabels(labels=ass.index, rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%y/%m'))

    plt.savefig("Xray-{}.png".format(ticker))
    plt.show()


def plot_buyP(ass,ticker,startDay=None,endDay=None,conf=None):
    '''
    此函数为买压、卖压训练效果画图函数
    :param ass:数据参数ass，为dataframe结构，包括
    :param ticker:
    :param startDay:
    :param endDay:
    :param conf:
    :return:
    '''
    pass


def plot_sellP(ass,ticker,startDay=None,endDay=None,conf=None):
    pass
