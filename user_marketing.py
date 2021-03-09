#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#chap 2.2 从数据库中读取数据
import pymysql

# 建立连接对象
connection =pymysql.connect(
    host='traindb.cookdata.cn',
    port=3306,
    user='raa_user',
    password='bigdata123',
    db='transactions',
charset='utf8mb4')

try:
    with connection.cursor() as cursor:
        # 查询数据表行数
        sql1 = "select count(*)from flow_data"
        # 选取表的前五行
        sql2 = "select * from flow_data limit 5"

        cursor.execute(sql1)
        rows = cursor.fetchall() 
        cursor.execute(sql2)
        head = cursor.fetchall() 

finally:
    connection.close()

# 查看数据的行列数
print('数据行数为:',rows,'\n前五行数据为:',head)


#对客户进行数据转换
import pandas as pd

# 将data转换为DataFrame格式
data = pd.DataFrame(list(data),columns=['user_id','payment','describe','unix_time'])#要传入一个list数据

print(data.head())


# In[ ]:


#3.2数据统计分析 
import pandas as pd

# 查看数据形状
rows, cols = data.shape

print("数据共有 %s 行，有 %s 列。" % (rows, cols))

# 查看数据的前五行
head = data.head()

print('\n',head)

# 查看数据的基本情况
data.info(null_counts=True) #null_counts=True则可以统计每一列非空的个数，这样就知道哪些列缺失值


# In[ ]:


#查看客户总数
import pandas as pd

# 计算客户个数
user_num = len(data['user_id'].unique())
print("客户总数为:",user_num)

# 计算客户交易次数
user_counts = data['user_id'].value_counts()#每条交易记录都存在，计数
print("每个客户的交易次数为:\n",user_counts)


# In[ ]:


#3.4交易时间异常值检测
import pandas as pd

# 书写正则表达式
#交易时间正常有10位，^是开始，$是结束标志
pattern = '^\d{10}$'

# 筛选异常值
outlier =data[ ~data['unix_time'].str.match( pattern,case=True)]#去匹配符合true的表达式，再取反

# 统计不同类型的异常值及其数量
outlier_counts = outlier['unix_time'].value_counts()

print(outlier_counts)


# In[ ]:


# 3.5交易异常时间处理
import pandas as pd

# 去掉交易时间为0的行
#loc,有些列的取值有选择
data = data.loc[data['unix_time']!=0]
#列和标量进行比较，因为pandas有广播特性
# 将异常值填补为正常值
data.loc[data['unix_time']=='14 3264000','unix_time'] = 1473264000

print(data.loc[data['unix_time'] == '14 3264000'])
print(data.loc[data['unix_time'] == 0])


# In[ ]:


#3.6交易金额异常值处理
import pandas as pd

# 查看交易金额为'\N'的行数
print("交易金额异常的记录共有%s行。" % (len(data[data['payment'] == '\\N'])))#bool数组匹配

# 去除交易金额为'\N'的行
data = data[data['payment']!='\\N']
#打印结果都是一个检测过程
print(data[data['payment'] == '\\N'


# In[ ]:


# 3.7交易附言缺失值处理
import pandas as pd

# 筛选describe中有无附言为空的行
describe_null =  data[data['describe'].isnull()]

print("交易附言为空的行共有%s条。"%  len(describe_null))
print(describe_null.head())


# In[ ]:


import pandas as pd

# 时间格式转换
#若没有该列会新增一列
data['pay_time'] = pd.to_datetime(data['unix_time'],unit='s')

# 时区转换
data['pay_time'] = data['pay_time']+pd.Timedelta(hours=8)#指明Hours

print(data.tail(5))


# In[ ]:


#3.9 量纲转化
import pandas as pd

# 将payment列标准化
data['payment'] =data['payment']/100

print(data.head())


# In[ ]:


#重复数据处理
import pandas as pd

# 检测重复值
#duplicated返回的是一个Series对象
duplicate_values =data[data.duplicated(subset=None,keep='first')]
print("重复数据有%s行。"% len(duplicate_values))

# 去掉重复值
data.drop_duplicates(subset=None,keep='first',inplace=True)
#很多函数中参数有inplace=ture,因为pandas很多函数是返回一个新对象，而不是在原对象上进行改变
print("处理之后，交易记录变为%s行。" % len(data))


# In[ ]:


#交易次数随时间的可视化分析
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))
#data.head()
# 绘制折线图
#plt=fig.add_subplot(2,2,1) 报错说fig.subplot没有show()函数
data['pay_time'].dt.date.value_counts().plot()#结果会自动排序

#Q?下面直接设置plt的属性，不是和包的名字重复了吗
# 设置图形标题
plt.title("不同时间的交易次数分布")

# 设置y轴标签
plt.ylabel("交易次数")

# 设置x轴标签
plt.xlabel("时间")
plt.show()


# In[ ]:


#交易金额随时间变化的可视化分析
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

# 绘制折线图
abs(data['payment']).groupby(data['pay_time'].dt.date).sum().plot()

# 设置图形标题
plt.title("不同时间的交易金额分布")
plt.xlabel("时间")
plt.ylabel("交易金额")
plt.show()


# In[ ]:


#交易有效时间的限定
import pandas as pd

# 时间限定
data=data[(data['pay_time']<=pd.Timestamp(2017,12,31))&(data['pay_time']>=pd.Timestamp(2016,7,1))]
#bools=data['pay_time']>=pd.Timestamp('2016-07-01')
#and data['pay_time']<=pd.Timestamp('2018-01-01')

print(data.shape)


# In[ ]:


# 4.5 每天24H的交易分布
import pandas as pd
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

# 绘制条形图
#小时无法自动识别时间大小进行排序，所以人为排序
data['pay_time'].dt.hour.value_counts().sort_index().plot.bar(color='orange',rot=360)
plt.title("每天24小时的交易次数分布")
plt.xlabel("小时")
plt.ylabel("交易次数")
plt.show()


# In[ ]:


#4.6 客户交易次数可视化分析
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 5))

# 绘制核密度图
sns.kdeplot(data.user_id.value_counts(),shade=True,legend=False)

# 设置x，y轴标签
plt.xlabel("交易次数")
plt.ylabel("频率")

# 设置图的标题
plt.title("客户交易次数分布")

plt.show()


# In[ ]:


#4.7 客户平均交易金额的可视化分析
import seaborn as sns
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16, 5))

# 绘制核密度图

sns.kdeplot(abs(data['payment']).groupby(data.user_id).mean(),shade=True,legend=False)
# 设置x，y轴标签
plt.xlabel("平均交易金额")
plt.ylabel("频率")

# 设置图的标题
plt.title("客户平均交易金额的分布")

plt.show()


# In[ ]:


#客户交易次数流入流出的可视化分析
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(16, 5))

# 定义和选取金额流入流出的交易记录
input_payment = data[data['payment']<0]
output_payment = data[data['payment']>0]

# 计算每个客户的流入流出次数
input_payment_count = input_payment['user_id'].value_counts()
output_payment_count =  output_payment['user_id'].value_counts()

# 绘制直方图
sns.distplot(input_payment_count,ax=ax1)
sns.distplot(output_payment_count,ax=ax2)
# 设置标题
ax1.set_title("客户交易的流入次数分布")
ax2.set_title("客户交易的流出次数分布")

plt.show()


# In[ ]:


'''
jieba.cut()`函数返回的结构是一个可迭代的`generator`，
可以使用`for`循环来获得分词后得到的每一个词语，
也可使用`.join()`函数将序列中的元素以指定的字符进行连接。
如上面的分词结果所示，将序列中的结果以`/`进行分隔，返回结果为字符串类型
'''
import pandas as pd
import jieba

# 数据采样
data = data.sample(20000,random_state = 22)

# 文本分词
data['describe_cutted'] = data['describe'].apply(lambda x:" ".join(jieba.cut(x))) 
#先cut，再用空格连接cut后的函数
#apply(参数里是函数)，对列进行批量处理

# 过滤停用词  
def del_stopwords(words):  
    output = ''  
    for word in words:  
        if word not in stopwords:  
            output += word
    return output

data['describe_cutted'] = data['describe_cutted'].apply(del_stopwords)

print(data.head())


# In[ ]:




