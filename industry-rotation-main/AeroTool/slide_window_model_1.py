'''
使用分位数训练，使用用卖压买压算出来的仓位y
'''

import gpytorch
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from sklearn.preprocessing import normalize
from AeroTool.AeroTool import AeroTool
from tool.date_map import date_map
from tool.normalize_data import normalize_test_data,normalize_train_data
from tool.compute_quantile import get_quantile,get_quantile_2,get_quantile_before
from tool.plot_tool import plot_nav
import matplotlib.ticker as ticker

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(5 / 2))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
        # self.rbf_model = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        # print("x shape:", x.shape)
        # print("mean_x shape:", mean_x.shape)
        # covar_x = self.covar_module(x)+self.rbf_model(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(SpectralMixtureGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4,ard_num_dims=7)
        self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self,x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SlideWindowModel():
    def __init__(self):
        self.x_data = pd.read_csv("../data/automobile_data.csv", encoding='utf-8', index_col=0)
        self.y_data = pd.read_csv("../data/801094.csv", encoding='utf-8', index_col=0)
        self.sell_model=None
        self.buy_model=None

    def predict_position(self,start_day,end_day):
        '''
        预测的仓位，预测的起始时间
        :param start_day: 开始时间
        :param end_day: 结束时间
        :return:
        '''
        pass


    def train_and_predict(self,train_start_day, train_end_day,test_start_day,test_end_day):
        '''
        指定数据用于训练模型和预测结果
        :param train_start_day: 训练数据开始的时间，int类型，如：20100212
        :param train_end_day: 训练数据结束时间.int类型，如：20100212
        :param test_start_day:测试数据开始时间，int类型，同上
        :param test_end_day:测试数据结束时间，int类型，同上
        :return: 返回训练好的预测结果
        '''
        print(train_start_day,train_end_day,test_start_day,test_end_day)
        x_data=self.x_data
        y_data=self.y_data
        train_x_data = x_data[x_data.index <= train_end_day].copy()
        train_x_data = train_x_data[train_x_data.index >= train_start_day]
        test_x_data = x_data[x_data.index >= test_start_day].copy()
        test_x_data=test_x_data[test_x_data.index< test_end_day]
        train_y_data = y_data[y_data.index <= train_end_day].copy()
        train_y_data = train_y_data[train_y_data.index >= train_start_day]
        test_y_data = y_data[y_data.index > train_start_day].copy()
        test_y_data = test_y_data[test_y_data.index <= train_end_day]

        before_day = 10
        test_x_data["1"] = get_quantile_2(train_x_data["1"], test_x_data["1"])
        test_x_data["2"] = get_quantile_2(train_x_data["2"], test_x_data["2"])
        test_x_data["3"] = get_quantile_2(train_x_data["3"], test_x_data["3"])
        test_x_data["4"] = get_quantile_2(train_x_data["4"], test_x_data["4"])
        test_x_data["5"] = get_quantile_2(train_x_data["5"], test_x_data["5"])
        test_x_data["6"] = get_quantile_2(train_x_data["6"], test_x_data["6"])
        test_x_data["7"] = get_quantile_2(train_x_data["7"], test_x_data["7"])
        test_x_data["8"] = get_quantile_2(train_x_data["8"], test_x_data["8"])
        # test_x_data["ratio"]=test_x_data["ratio"]
        # test_x_data["ratio"]=normalize_test_data(train_x_data["ratio"],test_x_data["ratio"])*2
        test_x_data["PE"] = get_quantile_before(test_x_data["PE"], before_day)
        test_x_data["ratio"] = get_quantile_before(test_x_data["ratio"], before_day)
        # test_x_data["PE"] = normalize(test_x_data[['PE']],axis=0,norm='max')*10
        test_x_data['volume'] = normalize(test_x_data[['volume']], axis=0, norm='max')

        train_x_data["1"] = get_quantile(train_x_data["1"])
        train_x_data["2"] = get_quantile(train_x_data["2"])
        train_x_data["3"] = get_quantile(train_x_data["3"])
        train_x_data["4"] = get_quantile(train_x_data["4"])
        train_x_data["5"] = get_quantile(train_x_data["5"])
        train_x_data["6"] = get_quantile(train_x_data["6"])
        train_x_data["7"] = get_quantile(train_x_data["7"])
        train_x_data["8"] = get_quantile(train_x_data["8"])
        train_x_data["PE"] = get_quantile_before(train_x_data["PE"], before_day)
        train_x_data["ratio"] = get_quantile_before(train_x_data["ratio"], before_day)
        # train_x_data["PE"]=normalize(train_x_data[["PE"]],norm='max',axis=0)
        train_x_data["volume"] = normalize(train_x_data[["volume"]], axis=0, norm='max')


        # "1","2","4","5","7","8","change_rate","volume"
        # train_x = train_x_data[["1","2","4","5","7","8","PE","close","ratio","change_rate"]]
        train_x = train_x_data[["1", "2", "4", "5", "7", "8","PE"]]
        test_x = test_x_data[["1", "2", "4", "5", "7", "8","PE"]]
        # print("train x:",train_x)
        # print("test x:",test_x)
        # test_x = test_x_data[["1","2","4","5","7","8","close","change_rate"]]

        train_y = train_y_data["ideal"]
        test_y = test_y_data["ideal"]

        train_y_sellP = train_y_data["sellP"]
        train_y_buyP = train_y_data["buyP"]
        test_y_sellP = test_y_data["sellP"]
        test_y_buyP = test_y_data["buyP"]

        # print(test_x)
        train_x = np.array(train_x)
        train_y = np.array(train_y)
        train_y_sellP = np.array(train_y_sellP)
        train_y_buyP = np.array(train_y_buyP)
        test_x = np.array(test_x)
        test_y = np.array(test_y)
        test_y_buyP = np.array(test_y_buyP)
        test_y_sellP = np.array(test_y_sellP)

        # train_x=normalize(train_x,axis=0,norm='max')
        # test_x=normalize(test_x,axis=0,norm='max')

        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)
        train_y_sellP = torch.Tensor(train_y_sellP)
        train_y_buyP = torch.Tensor(train_y_buyP)

        # initialize likelihood and model
        sell_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        buy_likelihood = gpytorch.likelihoods.GaussianLikelihood()

        sell_model = ExactGPModel(train_x, train_y_sellP, sell_likelihood)
        buy_model = ExactGPModel(train_x, train_y_buyP, buy_likelihood)

        buy_training_iter = 40
        sell_training_iter = 40

        # Use the adam optimizer
        sell_optimizer = torch.optim.Adam(sell_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        buy_optimizer = torch.optim.Adam(buy_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        sell_mll = gpytorch.mlls.ExactMarginalLogLikelihood(sell_likelihood, sell_model)
        buy_mll = gpytorch.mlls.ExactMarginalLogLikelihood(buy_likelihood, buy_model)

        # Find optimal model hyperparameters
        sell_model.train()
        sell_likelihood.train()
        buy_model.train()
        buy_likelihood.train()

        for i in range(sell_training_iter):
            # Zero gradients from previous iteration
            sell_optimizer.zero_grad()
            # Output from model
            output = sell_model(train_x)

            # Calc loss and backprop gradients
            # print("output:",output)
            # print("train_y：",train_y)
            loss = -sell_mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, sell_training_iter, loss.item(), 0,
                # sell_model.covar_module.base_kernel.lengthscale.item(),
                sell_model.likelihood.noise.item()
            ))
            sell_optimizer.step()
            if loss.item()<0.45:
                break

        for i in range(buy_training_iter):
            # Zero gradients from previous iteration
            buy_optimizer.zero_grad()
            # Output from model
            output = buy_model(train_x)

            # Calc loss and backprop gradients
            loss = -buy_mll(output, train_y)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, buy_training_iter, loss.item(), 0,
                # buy_model.covar_module.base_kernel.lengthscale.item(),
                buy_model.likelihood.noise.item()
            ))
            buy_optimizer.step()
            if loss.item()<0.5:
                break

        sell_model.eval()
        sell_likelihood.eval()
        buy_model.eval()
        buy_likelihood.eval()
        test_date = test_x_data.index
        test_x = torch.Tensor(test_x)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            y_test_sell_pre = sell_likelihood(sell_model(test_x)).mean.numpy()
            y_test_buy_pre = buy_likelihood(buy_model(test_x)).mean.numpy()
            y_test_short_weight = pd.Series(y_test_buy_pre, index=test_x_data.index)
            y_test_long_weight = pd.Series(y_test_sell_pre, index=test_x_data.index)
            conf = {"gradient_ratio": 1.2, "port_thres": 10, "granularity": 0.1}
            # print(AeroTool.combineWeight(y_test_short_weight, y_test_long_weight, conf))
            y_test_pre = AeroTool.combineWeight(y_test_short_weight, y_test_long_weight, conf)['Weight']
            y_test_pre = np.array(y_test_pre)
            del sell_model
            del buy_model
            return test_date,y_test_pre


    # int类型的日期向后推一定的月份
    def next_few_month(self,int_date,month_num=6):
        day = int_date % 100
        month = (int_date // 100) % 100
        month+=month_num
        year = int_date // 10000
        if month>12:
            month = month-12
            year+=1
        ret_date=year*10000+month*100+day
        return int(ret_date)

    def slide_window(self,days_num=126):
        '''
        使用滑动窗口预测未来一段时间的仓位，未来的时间默认为半年
        :param days_num: 预测未来的时间，默认为半年时间，即126天
        :return:
        '''
        train_start_day=20100301
        train_end_day=20130101
        test_start_day=20130401
        test_end_day=self.next_few_month(test_start_day,6)
        final_day=20201201

        # for i in range
        predict_arr=np.array([])
        test_date=np.array([])
        while test_end_day<final_day:
            test_date_sub,predict_sub_arr=self.train_and_predict(train_start_day,train_end_day,test_start_day,test_end_day)
            predict_arr=np.concatenate([predict_arr,predict_sub_arr])
            test_date=np.concatenate([test_date,test_date_sub])
            train_start_day=self.next_few_month(train_start_day,6)
            train_end_day=self.next_few_month(train_end_day,6)
            test_start_day=self.next_few_month(test_start_day,6)
            test_end_day=self.next_few_month(test_end_day,6)

        print("length:",len(predict_arr),len(test_date))
        print(test_date)
        ass = pd.DataFrame(columns=['tradingDay', 'ideal', 'logR', 'predict'])
        ass["predict"] = predict_arr
        print("test_date:",test_date)
        condition=np.array([x in test_date for x in self.y_data.index])
        # print(len(self.y_data[condition]))
        ass["tradingDay"] = np.array([str(int(x)) for x in test_date])
        ass["logR"] = np.array(self.y_data[condition]["logR"])
        ass["ideal"] = np.array(self.y_data[condition]["ideal"])
        print(ass)
        print("hello world slide window！")
        plot_nav(ass, "801094")



if __name__ == '__main__':
    slide_model=SlideWindowModel()
    slide_model.slide_window()


