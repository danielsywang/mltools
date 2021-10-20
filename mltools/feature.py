import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import (MinMaxScaler, PowerTransformer,
                                   QuantileTransformer, RobustScaler,
                                   StandardScaler)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 2021.09.14 Created by Daniel SY wang


def get_corr(data, method='pearson', threshold=0.7):
    """
    | 根据相关系数来选择特征,消除多重共线性的特征
    | Return the columns pair and correlation coefficient with correlation coefficient great threshold as a pandas Dataframe

    Parameters
    ----------
    data : df
        pandas Dataframe

    method : string
        'pearson','spearman','kendall'

    threshold : float
        default 0.7

    Returns
    -------
    df : df
        pandas Dataframe with null count and percent

    """

    corr = data.corr(method=method)
    x_list = list(corr.columns)
    y_list = x_list.copy()
    records = []
    for x in x_list:
        y_list.pop(0)
        if y_list:
            for y in y_list:
                if abs(corr.loc[x, y]) > threshold:
                    records.append([x, y, corr.loc[x, y]])
    df = pd.DataFrame(records, columns=['Feature1', 'Feature2', 'Corr'])
    df = df.sort_values('Feature1', ascending=True).reset_index(drop=True)
    return df


def get_VIF(data):
    """
    | 根据VIF来选择特征,消除多重共线性
    | detect the multicollinear features using the variance inflation factor 

    Parameters
    ----------
    data : df
        pandas Dataframe

    Returns
    -------
    vif_info : df
        pandas Dataframe with VIF value

    """

    vif_info = pd.DataFrame()
    vif_info['VIF'] = [variance_inflation_factor(
        data.values, i) for i in range(data.shape[1])]
    vif_info['Column'] = data.columns
    vif_info = vif_info.sort_values(
        'VIF', ascending=True).reset_index(drop=True)
    return vif_info


def normal_test(data, method='jb'):
    """
    检验数据是否服从正态分布,三种检验方案都是用于大数据分析\n

    | 提出假设：x从正态分布。
    | P值>指定水平0.05,接受原假设，可以认为样本数据在5%的显著水平下服从正态分布,以'normal'表示.
    | P值<指定水平0.05,拒绝原假设，认为样本数据在5%的显著水平下不服从正态分布,以'-'表示. 

    有三种备选检测方案：

    stats.jarque_bera(data)

    stats.kstest(rvs, cdf, args=(), N=20, alternative=’two_sided’, mode=’approx’, \*\*kwds)
    对于正态性检验，我们只需要设置三个参数即可：
    rvs：待检验的数据
    cdf：检验方法，这里我们设置为‘norm’，即正态性检验
    alternative：默认为双尾检验，可以设置为‘less’或‘greater’作单尾检验。

    scipy.stats.normaltest(a, axis=0, nan_policy=’propagate’)
    这里的三个参数都有必要看一下：
    a：待检验的数据
    axis：默认为0，表示在0轴上检验，即对数据的每一行做正态性检验，我们可以设置为 axis=None 来对整个数据做检验
    nan_policy：当输入的数据中有空值时的处理办法。默认为 ‘propagate’，返回空值；
    设置为 ‘raise’ 时，抛出错误；设置为 ‘omit’ 时，在计算中忽略空值。    

    Parameters
    ----------
    data : df
        pandas Dataframe

    method : string
        'jb'（default）, 'ks' , 'norm'

    Returns
    -------
    data_test : df
        pandas Dataframe 

    """
    col_test = []
    for col in data.columns:
        s = data[col]
        s.replace([np.inf, -np.inf], np.nan, inplace=True)
        s = s[s.notnull()]

        if method == 'jb':
            _, pval = stats.jarque_bera(s)
        elif method == 'ks':
            _, pval = stats.kstest(s, 'norm')
        elif method == 'norm':
            _, pval = stats.normaltest(s, axis=None)

        if pval < 0.05:
            result = '-'
        else:
            result = 'Normal'
        col_test.append(result)
    data_test = pd.DataFrame(data=col_test, index=data.columns)
    return data_test


def transformer(data, trans_method_list='all', test_method='jb'):
    """
    先对数据集进行转换，然后检验转换后的数据是否服从正态分布\n

    Parameters
    ----------
    data : df
        pandas Dataframe

    trans_method_list : list
        a list of 'sqrt','cbrt','log1p','reciprocal',
        'square','cube','expm1','minmax','zscore',
        'robust','quant','box','yeo'.
        user can choice some transfomer method as a list.
        such as: list of 'minmax','zscore','robust','box','yeo'.
        default includes all method.    

    test_method : string
        'jb'（default）, 'ks' , 'norm'

    Returns
    -------
    result : df
        pandas Dataframe

    """
    if trans_method_list == 'all':
        trans_method_list = ['sqrt', 'cbrt', 'log1p', 'reciprocal', 'square',
                             'cube', 'expm1', 'minmax', 'zscore', 'robust', 
                             'quant', 'box', 'yeo']

    result = normal_test(data, method=test_method)
    for trans_method in trans_method_list:
        if trans_method == 'sqrt':
            temp_data = np.sqrt(data)
        elif trans_method == 'cbrt':
            temp_data = np.cbrt(data)
        elif trans_method == 'log1p':
            temp_data = np.log1p(data)
        elif trans_method == 'reciprocal':
            temp_data = np.reciprocal(data)
        elif trans_method == 'square':
            temp_data = np.square(data)
        elif trans_method == 'cube':
            temp_data = np.power(data, 3)
        elif trans_method == 'expm1':
            temp_data = np.expm1(data)
        elif trans_method == 'minmax':
            trans = MinMaxScaler()
        elif trans_method == 'zscore':
            trans = StandardScaler()
        elif trans_method == 'robust':
            trans = RobustScaler()
        elif trans_method == 'quant':
            trans = QuantileTransformer(
                n_quantiles=500, output_distribution='normal', random_state=100)
        elif trans_method == 'box':
            # The Box-Cox transformation can only be applied to strictly positive data
            trans = PowerTransformer(method='box-cox')
        elif trans_method == 'yeo':
            trans = PowerTransformer(method='yeo-johnson')

        if trans_method in ['minmax', 'zscore', 'robust', 'quant', 'box', 'yeo']:
            temp_data = pd.DataFrame(
                trans.fit_transform(data), columns=data.columns)

        temp = normal_test(temp_data, method=test_method)
        result = pd.concat([result, temp], axis=1)
    result.columns = ['orginal']+trans_method_list
    return result
