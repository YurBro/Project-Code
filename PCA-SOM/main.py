#/usr/env/bin python
# -*- coding:utf-8 -*-
# author: Maxwell_yurz
# date: 2021/06/25

# insert library
import numpy as np
import pandas as pd



# convriance
import seaborn as sns;sns.set(color_codes=True)

# pca
from pylab import *
from scipy.stats import f


import xlrd


# 设置迭代器进度条
from tqdm import tqdm


# 数据归一化
from sklearn import preprocessing

import plotly.graph_objects as go



# 导入数据
# 传入数据成numpy的矩阵格式
def loaddata(datafile, num_name):
    df = pd.read_excel(datafile, sheet_name=num_name)

    return df


'''-------------------------- 1.PCA部分 --------------------------'''

# 相关性分析
def covriance_o(d, layer):
    X = d.corr()
    print('第%r层协方差矩阵如下：' % layer)
    print(X)
    # plt.figure(figsize=(10, 5))
    # plt.subplots_adjust(left=0.09, right=1, wspace=0.25, hspace=0.25, bottom=0.13, top=0.91)
    sns.heatmap(X, square=True, annot=True)
    plt.title("Correlation matrix-" + layer)
    # plt.savefig("Correlation matrix-" + layer + '.png', dpi=300)  # 指定分辨率
    plt.show()



"""
参数：
	- XMat：传入的是一个numpy的矩阵格式，行表示样本数，列表示特征    
	- k：表示取前k个特征值对应的特征向量
返回值：
	- finalData：参数一指的是返回的低维矩阵，对应于输入参数二
	- reconData：参数二对应的是移动坐标轴后的矩阵
"""


# 1.2成分矩阵 -- 项目需求：出.csv表格文件

def component_score_matrix(train_file_name, test_file_name, num_name):

    train_data = pd.read_excel(train_file_name, sheet_name=num_name)    # 导入训练数据
    test_data = pd.read_excel(test_file_name, sheet_name=num_name)     # 导入测试数据



    # *****************使用pandas方法读取样本数据功能模块（结束）*********************
    m = train_data.shape[1];  # 获取数据表格的列数
    n = train_data.shape[0];  # 获取数据表格的行数
    # ********************************* 归一化 *********************************
    data = train_data.iloc[:, :].copy().values
    print('切片数据集：', data)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(data)
    print('数据归一化后的数据集：', X_train_minmax)
    train_data = X_train_minmax
    test_data = X_train_minmax
    # ******************数据标准化处理（开始）*********************
    S_mean = np.mean(train_data, axis=0)  # 健康数据矩阵的列均值
    S_mean = np.array(S_mean)  # 健康数据的列均值，narry数据类型
    S_var = np.std(train_data, ddof=1);  # 健康数据矩阵的列方差,默认ddof=0表示对正态分布变量的方差的最大似然估计，ddof=1提供了对无限总体样本的方差的无偏估计（与Matlab一致）
    # S_var[S_var == 0.0] = 0.0000000000000001  # 将集合S_var中的0替换为0.0000000000000001
    S_var = np.array(S_var)  # 健康数据的列方差，narry数据类型
    train_data -= S_mean  # 求取矩阵X的均值
    train_data /= S_var  # 求取矩阵X的方差
    train_data = np.where(train_data < 4.0e+11, train_data, 0.0)  # 把标准化后的矩阵X中的0替换为0.0000000000000001
    X_new = train_data;  # 求得标准化处理后的矩阵X_new
    # ******************求矩阵Y的协方差矩阵Z*********************
    X_new = np.transpose(X_new);  # 对矩阵进行转秩操作
    Z = np.dot(X_new, train_data / (n - 1))  # 求取协方差矩阵Z
    # ******************计算协方差矩阵Z的特征值和特征向量*********************
    a, b = np.linalg.eig(Z)  ##特征值赋值给a，对应特征向量赋值给b
    lambda1 = sorted(a, reverse=True)  # 特征值从大到小排序
    lambda_i = [round(i, 3) for i in lambda1]  # 保留三位小数
    print('lambda特征值由大到小排列：', lambda_i)
    # 计算方差百分比
    sum_given = 0  # 设置初值为0
    sum_given = sum(lambda_i)
    variance_hud = []  # 设置存放方差百分比的矩阵
    for i in tqdm(range(m)):
        if i <= m:
            variance_hud.append(lambda_i[i] / sum_given)
        else:
            break
    variance_hud = [round(i, 3) for i in variance_hud]  # 保留三位小数
    print('方差百分比从大到小排序：', variance_hud)

    # 累计贡献率
    leiji_1 = []
    new_value = 0
    for i in tqdm(range(0, m)):
        if i <= m:
            new_value = new_value + variance_hud[i]
            leiji_1.append(new_value)
        else:
            break

    print('累计贡献率：', leiji_1)

    # ******************主元个数选取 *********************
    totalvar = 0   # 累计贡献率，初值0
    for i in tqdm(range(m)):
        totalvar = totalvar + lambda1[i] / sum(a)  # 累计贡献率，初值0
        if totalvar >= 0.85:
            k = i + 1  # 确定主元个数
            break  # 跳出for循环
    PCnum = k  # 选取的主元个数
    PC = np.eye(m, k)  # 定义一个矩阵，用于存放选取主元的特征向量
    for j in tqdm(range(k)):
        wt = a.tolist().index(lambda1[j])  # 查找排序完成的第j个特征值在没排序特征值里的位置。
        PC[:, j:j + 1] = b[:, wt:wt + 1]  # 提取的特征值对应的特征向量
    print('成分矩阵：', PC)
    print('贡献率85%以上的主元个数为：', k)

    df_cfjz = pd.DataFrame(PC)

    # ******************根据建模数据求取 T2 阈值限 *********************
    # ******************置信度 = (1-a)% =（1-0.05）%=95% *************
    F = f.ppf(1 - 0.05, k, n - 1)  # F分布临界值
    T2 = k * (n - 1) * F / (n - k)  # T2求取
    # ****************** 健康数据的 SPE 阈值限求解  *********************
    ST1 = 0  # 对应SPE公式中的角1初值
    ST2 = 0  # 对应SPE公式中的角2初值
    ST3 = 0  # 对应SPE公式中的角3初值
    for i in range(k - 1, m):
        ST1 = ST1 + lambda1[i]  # 对应SPE公式中的角1
        ST2 = ST2 + lambda1[i] * lambda1[i]  # 对应SPE公式中的角2
        ST3 = ST3 + lambda1[i] * lambda1[i] * lambda1[i]  # 对应SPE公式中的角3
    h0 = 1 - 2 * ST1 * ST3 / (3 * pow(ST2, 2))
    Ca = 1.6449
    SPE = ST1 * pow(Ca * pow(2 * ST2 * pow(h0, 2), 0.5) / ST1 + 1 + ST2 * h0 * (h0 - 1) / pow(ST1, 2),
                    1 / h0)  # 健康数据SPE计算
    # ******************测试样本数据*********************
    m1 = test_data.shape[1];  # 获取数据表格的列数
    n1 = test_data.shape[0];  # 获取数据表格的行数
    test_data = np.array(test_data)  # 将DataFrame数据烈性转化为ndarray类型，使得数据矩阵与Matlab操作一样。
    I = np.eye(m)  # 产生m*m的单位矩阵
    PC1 = np.transpose(PC)  # PC的转秩
    SPEa = np.arange(n1).reshape(1, n1)  # 定义测试数据的SPE矩阵,为正数矩阵
    SPEa = np.double(SPEa)  # 将正数矩阵，转化为双精度数据矩阵
    TT2a = np.arange(n1).reshape(1, n1)  # 定义测试数据的T2矩阵,为正数矩阵
    TT2a = np.double(TT2a)  # 将正数矩阵，转化为双精度数据矩阵
    DL = np.diag(lambda1[0:k])  # 特征值组成的对角矩阵
    DLi = np.linalg.inv(DL)  # 特征值组成的对角矩阵的逆矩阵
    # ******************绘制结果 *********************
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 在图形中显示汉字
    for i in range(n1):
        xnew = (test_data[i, :] - S_mean) / S_var;  # 对应 Matlab程序：xnew=(Data2(i,1:m)-S_mean)./S_var;
        # 以下是实现Matlb程序：  err(1,i)=xnew*(eye(14)-PC*PC')*xnew';
        xnew1 = np.transpose(xnew)  # xnew的转秩
        PC1 = np.transpose(PC)  # PC的转秩
        XPC = np.dot(xnew, PC)  # 矩阵xnew与PC相乘
        XPCPC1 = np.dot(XPC, PC1)  # 矩阵XPC与PC1相乘
        XXPCPC1 = xnew - XPCPC1  # 矩阵xnew减去XPCPC1
        SPEa[0, i] = np.dot(XXPCPC1, XXPCPC1)  # 矩阵XXPCPC1与XXPCPC1相乘
        XPi = np.dot(XPC, DLi)  # 矩阵XPC与DLi相乘
        XPiP = np.dot(XPi, PC1)  # 矩阵XPi与PC1相乘
        TT2a[0, i] = np.dot(XPiP, xnew1)  # 矩阵XPiP与xnew1相乘
    Sampling = r_[0.:n1]  # 产生的序列值式0到n1
    SPE1 = SPE * ones((1, n1))  # 产生SPE数值相同的矩阵
    print('spe统计量的值：', SPEa)
    # df_spe = pd.DataFrame(SPEa.T)
    new_SPE = SPEa.T
    # df_spe.to_csv('SPE值.csv')     # 将SPE值保存成.csv
    T21 = T2 * ones((1, n1))  # 产生T2数值相同的矩阵
    print('t2统计量的值：', TT2a)
    # df_T2 = pd.DataFrame(TT2a.T)
    new_TT = TT2a.T
    # df_T2.to_csv('T2值.csv')       # 将T2值保存成.csv
    return new_SPE, new_TT, Sampling, TT2a, T21, SPEa, SPE1, n1, T2, SPE, m, variance_hud, leiji_1, df_cfjz

# 可视化T2和SPE
def graph_TT_SPE(Sampling, TT2a, T21, SPEa, SPE1, n1, T2, SPE, layer):
    ###########################################################################################
    figure(1)  # 画的第一张图
    plot(Sampling, TT2a[0, :], '*-', Sampling, T21[0, :], 'r-')  # 绘制出测试数据SPEa的数据集合，和健康数据训练得到的SPE阈值限
    xlabel('sample points')  # 给X轴加标注
    ylabel('T^2')  # 给Y轴加标注
    legend(['T^2 value', 'T^2 limit'])  # 为绘制出的图形线条添加标签注明
    title("T^2 statistic" + layer)  # 绘制的图形主题为“SPE统计量”
    # show()#显示绘制的图形
    figure(2)
    plot(Sampling, SPEa[0, :], '*-', Sampling, SPE1[0, :], 'r-')  # 绘制出测试数据TT2a的数据集合，和健康数据训练得到的T2阈值限
    xlabel('sample points')  # 给X轴加标注
    ylabel('SPE')  # 给Y轴加标注
    legend(['SPE value', 'SPE limit'])  # 为绘制出的图形线条添加标签注明
    title("SPE statistic" + layer)  # 绘制的图形主题为“SPE统计量”
    show()  # 显示绘制的图形
    #########构建循环输出######################################################################
    # 循环对象TT2a,SPEa,循环基线T2,SPE
    sum1 = 0
    for ij in range(n1):  # 对测试样本个数进行循环
        if ((TT2a[0, ij] <= T2) & (SPEa[0, ij] <= SPE)):  # 判断各个值是否小于阈值线
            TT2a[0, ij] = 0  # 将小于阈值线的样本点位置上的数置为0
            SPEa[0, ij] = 0  # 将小于阈值线的样本点位置上的数置为0
        else:
            TT2a[0, ij] = 1  # 将小于阈值线的样本点位置上的数置为1
            SPEa[0, ij] = 1  # 将小于阈值线的样本点位置上的数置为1
            sum1 += 1
            # print(i)#输出有故障的样本点
    print(sum1)
    ###########################################################################################
    d1 = pd.DataFrame(TT2a.T)
    d1['label'] = d1[0]
    d1.drop(0, axis=1, inplace=True)
    d1.to_csv('label.csv', index=False)
    print(d1.sum())
    print(SPEa)
    ##########################################################################################

    # 设置返回值


'''---------------------------- 3.SOM模型 ------------------------'''
# MiniSom lib --项目需求：1.神经元聚类图、2.流场特征（此图由甲方觉得再做修改）






'''---------------------------- pyrcharts或matplot绘图-------------'''
# pyecharts绘图
# 热力图





# matplot绘制柱状图
def plotcolumn(data1, num_x1, layer_1):
    num_list = data1
    plt.title('Contribution Rate-' + layer_1, fontsize=18)  # 标题，并设定字号大小
    plt.xlabel(u'Principal component', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'Percentage', fontsize=14)  # 设置y轴，并设定字号大小
    for a, b in zip(num_x1, num_list):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=12, color='orange')
    plt.bar(num_x1, num_list)
    plt.grid(linestyle='-.', axis='y')
    # plt.savefig('Contribution Rate-' + layer_1 + '.png', dpi=300)
    print('第%r层主元贡献率图保存成功！！！' % layer_1)
    plt.show()



# matplot折线图
def plotzx(data2, num_x2, layer_2):
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False
    num_list = data2
    plt.figure(figsize=(10, 5))  # 设置画布的尺寸
    for a, b in zip(num_x2, num_list):
        plt.text(a, b+0.01, b, ha='center', va='bottom', fontsize=12, color='orange')
    plt.title('Percentage of variance-'+layer_2, fontsize=20)  # 标题，并设定字号大小
    plt.xlabel(u'Principal component', fontsize=14)  # 设置x轴，并设定字号大小
    plt.ylabel(u'Percentage', fontsize=14)  # 设置y轴，并设定字号大小

    # color：颜色，linewidth：线宽，linestyle：线条类型，label：图例，marker：数据点的类型
    plt.plot(num_x2, num_list, color="deeppink", linewidth=2, linestyle=':', label='Percentage', marker='o')

    plt.legend(loc='best')  # 图例展示位置，数字代表第几象限
    # plt.savefig('Percentage of variance-'+layer_2 + '.png', dpi=300)
    print('第%r层碎石图保存成功！！！' % layer_2)
    plt.show()  # 显示图像



if __name__ == "__main__":

    # 导入原始数据
    datafile = "GD523INITA1_0.xls"

    # finalData, reconMat, featva, featvv, covriance, lambda_1, variance_1, leiji_1 = pca(XMat, k)


    # 多次运算得出T2和SPE值，存入一个.csv文件里，然后再绘制3D地形图---surface-wave
    # 通过循环多次访问原始数据表的表单，进行多次运算
    # for i in range():
    workbook = xlrd.open_workbook(datafile)
    sheets = workbook.sheet_names()
    SheetNames = []
    for sheetname in tqdm(sheets):
        print("表格的名称是：", sheetname)
        SheetNames.append(sheetname)
    print('原始数据表的表单名称为：', SheetNames)
    num_n = pd.DataFrame(SheetNames).shape[0]  # 获取表单的个数
    print('表单的个数为：', num_n)

    non_matrix1 = []  # 设置一个空矩阵,用于存放T2的值
    non_matrix2 = []  # 设置一个空矩阵,用于存放SPE的值

    # 循环
    for ii in tqdm(range(num_n)):
        if ii < num_n:
            print('程序目前处在第%r层数：' % SheetNames[ii])
            XMat2 = loaddata(datafile, num_name=SheetNames[ii])  # 返回得到浮点型矩阵
            # print('输入的矩阵为：', XMat)

            # 计算协方差矩阵，绘制相关性分析热力图
            covriance_o(XMat2, layer=SheetNames[ii])  # 协方差矩阵计算及出图 <success>


            # PCA初始化传值
            SPE_T, TT_T, sampling_n, TT2a_n, T21_n, SPEa_n, SPE1_n, n1_n, T2_n, SPE_n, m_n, variance_1, leiji_1, chengfenjz = component_score_matrix(
                train_file_name=datafile, test_file_name=datafile, num_name=SheetNames[ii])

           #  保存成分矩阵
            chengfenjz.to_csv('成分矩阵-'+SheetNames[ii]+'.csv', header='Value')
            print('第%r层成分矩阵储存成功！！！' % SheetNames[ii])

            # 单独将每次的T2和SPE存入单个.csv中
            # non_matrix2.append(SPE_T)     # 以下有问题

            print('转置后单独一层的SPE值：', SPE_T)
            SPEE = pd.DataFrame(SPE_T, columns=['Value'+SheetNames[ii]])
            # SPEE.iloc[0, 0] = "Value" + SheetNames[ii] # 修改列表名
            SPEE.to_csv('SPE_'+SheetNames[ii]+'_ValueSet.csv')
            print('第%r层SPE值储存成功！！！' % SheetNames[ii])
            # print('目前存放的SPE数据集为：', non_matrix2)


            # non_matrix1.append(TT_T)
            print('转置后单独一层T2值：', TT_T)
            Ttwo = pd.DataFrame(TT_T, columns=["Value" + SheetNames[ii]])
            # Ttwo.iloc[0, 0] = "Value" + SheetNames[ii]  # 修改列表名
            Ttwo.to_csv('T2_'+SheetNames[ii]+'_ValueSet.csv')
            print('第%r层T2值储存成功！！！' % SheetNames[ii])
            # print('目前存放的T2数据集为：', non_matrix1)
            # 将T2和SPE值存入一个矩阵内，保存.csv

            num_x = []
            for i in range(1, m_n + 1):
                num_x.append(i)  # 获取维度长度
            print('贡献率的x轴数值：', list(num_x))

            # 显示主元累计贡献率的柱状图
            plotcolumn(data1=leiji_1, num_x1=num_x, layer_1=SheetNames[ii])
            # 显示主元贡献率碎石图
            plotzx(data2=variance_1, num_x2=num_x, layer_2=SheetNames[ii])

            # T2和SPE图
            # graph_TT_SPE(Sampling=sampling_n, TT2a=TT2a_n, T21=T21_n, SPEa=SPEa_n, SPE1=SPE1_n, n1=n1_n, T2=T2_n, SPE=SPE_n, layer=SheetNames[ii])




        else:
            break

    # 将单独的T2和SPE文件分别合并成一个表格

    # 操作合并T2数据csv
    # 读取表格数据
    print('开始合并表格数据！')
    df_T2_first = pd.DataFrame(pd.read_csv('T2_' + SheetNames[0] + '_ValueSet.csv'))   # 首先读取第一个T2的表格, index_col=0 消除unnamed：0列
    time.sleep(1)  # 延迟1s

    # 操作合并SPE数据csv
    # 读取表格数据
    df_SPE_first = pd.DataFrame(pd.read_csv('SPE_' + SheetNames[0] + '_ValueSet.csv'))
    time.sleep(1)  # 延迟1s

    # 首先读取第一个SPE的表格, index_col=0 消除unnamed：0列
    # 然后，将剩下的T2表格都添加到第一个T2表格中去(SPE 同理)
    for nf in tqdm(range(1, num_n)):
        if nf < num_n:
            df_x = pd.read_csv('T2_' + SheetNames[nf] + '_ValueSet.csv')  # 从第二个T2表格开始读取
            df_xx = pd.DataFrame(df_x)
            print('目前读取的第%r层T2数据：' % SheetNames[nf])
            print(df_xx)

            df_SPEE = pd.DataFrame(pd.read_csv('SPE_' + SheetNames[nf] + '_ValueSet.csv'))
            print('目前读取的第%r层SPE数据：' % SheetNames[nf])
            print(df_SPEE)

            df_T2_first = df_T2_first.merge(df_xx)   # 将每次读取的T2增加到最右边
            df_SPE_first = df_SPE_first.merge(df_SPEE)  # 将每次读取的T2增加到最右边

            # data_base = pd.merge(df_T2_first, df_x, on=['Value'+SheetNames[nf]], how='left')  # 将csv数据表左连接
        else:
            # 设置一个完成提示
            print('T2和SPE数据表合并失败！～')
            time.sleep(2)  # 延迟2s
            break

    print('T2数据表合并完毕！！！')
    data_base1 = df_T2_first
    time.sleep(2)  # 延迟2s
    print("新的T2数据表为：")
    print(data_base1)
    time.sleep(2)  # 延迟2s
    # 写入全新的T2表格中
    data_base1.to_csv('T2_All_Value.csv', encoding='gbk')
    print('合并的T2数据保存成功！！！')

    print('SPE数据表合并完毕！！！')
    data_base2 = df_SPE_first
    time.sleep(2)  # 延迟2s

    # 输出合并后的data数据表内容
    print("新的SPE数据表为：")
    print(data_base2)

    time.sleep(2)  # 延迟2s
    data_base2.to_csv('SPE_All_Value.csv', encoding='gbk')
    print('合并的SPE数据保存成功！！！')


print('<-开始二维制图->')
print('制图中...')
print('...')

# 绘制T2三维图
'''---------------------------T2二维热力图function--------------------------'''

# 关于取xyz的值做矩阵
df_all_T2 = pd.read_csv('T2_All_Value.csv', index_col=0)  # index_col=0用于消除unnamed:0列
# 读取T2数据集的行列数
q1 = df_all_T2.shape[1]  # 列数
r1 = df_all_T2.shape[0]  # 行数
print('T2有%r行，%r列。' % (r1, q1))
# 读取x和y轴数据
df_x_y = pd.read_csv('x_y_axis.csv')
# 获取表格的行列数q,r
q2 = df_x_y.shape[1]  # 列数
r2 = df_x_y.shape[0]  # 行数

print('x_y有%r行，%r列。' % (r2, q2))

list_ix = []
list_iy = []
list_iz = []
one_data = []
all_data = []

# for xxx in range(1, q1+1):
#     if xxx < q1:
#         print('第%r层输出开始！' % df_all_T2.columns[xxx])

def choose_layer(layers):
    for xx in range(r2 + 1):
        if xx < r2:
            IIx = df_x_y.iloc[xx, 0]
            IIy = df_x_y.iloc[xx, 1]
            IIz = df_all_T2.iloc[xx, layers]

            # print('[x, y, z] = [%r, %r, %r]' % (IIx, IIy, IIz))
            # 将每个坐标轴存入一个list中
            list_ix.append(IIx)
            list_iy.append(IIy)
            list_iz.append(IIz)
    # print('本次的数据：', one_data)
    mmx = list_ix
    mmy = list_iy
    mmz = list_iz

    return mmx, mmy, mmz


def get_data(mmxx, mmyy, mmzz):


    df = pd.DataFrame(data=[v for v in zip(mmxx, mmyy, mmzz)], columns=['x', 'y', 'Value'])
    return df


if __name__ == '__main__':
    # SIZE = 100
    for i in range(1, q1):

        mmx, mmy, mmz = choose_layer(i)
        df = get_data(mmxx=mmx, mmyy=mmy, mmzz=mmz)

        layout = go.Layout(
            # plot_bgcolor='red',  # 图背景颜色
            paper_bgcolor='white',  # 图像背景颜色
            autosize=True,
            # width=2000,
            # height=1200,
            title=str(i) + '-热力图',
            titlefont=dict(size=30, color='gray'),

            # 图例相对于左下角的位置
            legend=dict(
                x=0.02,
                y=0.02
            ),

            # x轴的刻度和标签
            xaxis=dict(title='x坐标轴数据',  # 设置坐标轴的标签
                       titlefont=dict(color='red', size=20),
                       tickfont=dict(color='blue', size=18, ),
                       tickangle=45,  # 刻度旋转的角度
                       showticklabels=True,  # 是否显示坐标轴
                       # 刻度的范围及刻度
                       # autorange=False,
                       # range=[0, 100],
                       # type='linear',
                       ),

            # y轴的刻度和标签
            yaxis=dict(title='y坐标轴数据',  # 坐标轴的标签
                       titlefont=dict(color='blue', size=18),  # 坐标轴标签的字体及颜色
                       tickfont=dict(color='green', size=20, ),  # 刻度的字体大小及颜色
                       showticklabels=True,  # 设置是否显示刻度
                       tickangle=-45,
                       # 设置刻度的范围及刻度
                       autorange=True,
                       # range=[0, 100],
                       # type='linear',
                       ),
        )

        fig = go.Figure(data=go.Heatmap(
            showlegend=True,
            name='Value',
            x=df['x'],
            y=df['y'],
            z=df['Value'],
            type='heatmap',
        ),
            layout=layout
        )

        fig.update_layout(margin=dict(t=100, r=150, b=100, l=100), autosize=True)

        fig.show()



# End_node
time.sleep(3)  # 延迟3s,防跑飞
print('Success Work!')














