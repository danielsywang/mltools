import pandas as pd

# 2021.09.14 Created by Daniel SY wang


def null_count(data):
    """
    空缺值统计

    Return the columns's null count and percent with null count great zero as a pandas Dataframe

    Parameters
    ----------
    data : df
        pandas Dataframe


    Returns
    -------
    result : df
        pandas Dataframe with null count and percent

    """

    result = pd.concat([data.isnull().sum(), data.isnull().mean()*100], axis=1)
    result = result.rename(index=str, columns={
                           0: 'total_missing', 1: 'missing_percent'})

    # result.round(2)pd小数位数:2位
    col = result.columns[0]
    result = result[result[col] > 0].sort_values(
        by=col, ascending=False).round(2)
    return result


def unique_count(data):
    """
    唯一值检查\n
    Return the columns's with one value as a pandas Dataframe

    Parameters
    ----------
    data : df
        pandas Dataframe


    Returns
    -------
    unique_df : df
        pandas Dataframe with one value 

    """
    unique_stats = pd.DataFrame(data.nunique()).rename(columns={0: 'nunique'})

    # 不需要索引名
    # unique_stats.index.set_names('feature',inplace=True)
    col = unique_stats.columns[0]
    unique_df = unique_stats[unique_stats[col] == 1]

    return unique_df


def zero_count(data):
    """
    每列0值统计\n
    Return the columns's zero count and percent with zero count great zero as a pandas Dataframe

    Parameters
    ----------
    data : df 
        pandas Dataframe


    Returns
    -------
    zero_df : df
        pandas Dataframe with zero count and percent. 

    """

    zero_df = pd.DataFrame(data[data == 0].count()).sort_values(
        by=0, ascending=False).rename(columns={0: 'zero_count'})
    zero_df['zero_percent'] = (zero_df['zero_count']/len(data)*100).round(2)
    col = zero_df.columns[0]
    zero_df = zero_df[zero_df[col] > 0]

    return zero_df


def negative_count(data):
    """
    每列负值统计\n
    Return the columns's negative value count and percent with negative count great zero as a pandas Dataframe

    Parameters
    ----------
    data : df
        pandas Dataframe


    Returns
    -------
    negative_df : df
        pandas Dataframe with negative count and percent 

    """

    negative_df = pd.DataFrame(data[data < 0].count()).sort_values(
        0, ascending=False).rename(columns={0: 'negative_count'})
    negative_df['negative_percent'] = (
        negative_df['negative_count']/len(data)*100).round(2)
    col = negative_df.columns[0]
    negative_df = negative_df[negative_df[col] > 0]

    return negative_df


def outlier_detect(data, method='IQR', threshold=1.5):
    """
    每列离群点统计
        outlier detection by Mean and Standard Deviation Method.
        If a value is a certain number(called threshold) of standard deviations away
        from the mean, that data point is identified as an outlier. 
        The more extreme the outlier, the more the standard deviation is affected.

        outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
        calculate the IQR ( 75th quantile - 25th quantile) 
        and the 25th 75th quantile. 
        Any value beyond is regarded as outliers.
        upper bound = 75th quantile + （IQR * threshold）
        lower bound = 25th quantile - （IQR * threshold）   

    Parameters
    ----------
    data : df
        pandas Dataframe

    method : string
        'IQR','mean_std'

    threshold : float
        for IQR Default is 1.5,extreme 3. for mean_std is 3.

    Returns
    -------
    df_count : df
        pandas Dataframe with outlier count and percent 

    outlier_df : df
        outlier values of each columns
    """

    if method == 'mean_std':
        upper_fence = data.mean()+threshold*data.std()
        lower_fence = data.mean()-threshold*data.std()
    elif method == 'IQR':
        qt75 = data.quantile(0.75)
        qt25 = data.quantile(0.25)
        IQR = qt75-qt25
        upper_fence = qt75+threshold*IQR
        lower_fence = qt25-threshold*IQR

    upper_count = data[data > upper_fence].count()
    lower_count = data[data < lower_fence].count()

    outlier_count = pd.concat(
        [lower_fence, upper_fence, lower_count, upper_count], axis=1)
    outlier_count.columns = ['lower_fence',
                             'upper_fence', 'lower_outlier', 'upper_outlier']
    outlier_count['total_outlier'] = outlier_count['lower_outlier'] + \
        outlier_count['upper_outlier']
    outlier_count['total_percent'] = outlier_count['total_outlier'] / \
        len(data)*100
    df_count = outlier_count.sort_values(
        by='total_outlier', ascending=False).round(2)

    outlier_df = data[(data > upper_fence) | (
        data < lower_fence)].dropna(how='all')
    return df_count, outlier_df


def delete_extreme_outlier(data, method='IQR', threshold=3):
    """
    |   delete extreme outlier
    |   outlier detected by Mean and Standard Deviation Method.
    |   If a value is a certain number(called threshold) of standard deviations away
    |   from the mean, that data point is identified as an outlier. 
        The more extreme outlier, the more the standard deviation is affected.

    Outlier detection by Interquartile Ranges Rule, also known as Tukey's test. 
    calculate the IQR ( 75th quantile - 25th quantile) 
    and the 25th 75th quantile. 
    Any value beyond is regarded as outliers.
    upper_bound = 75th quantile + （IQR * threshold）
    lower_bound = 25th quantile - （IQR * threshold）   

    Parameters
    ----------
    data : df
        pandas Dataframe

    method : string
        'IQR','mean_std'

    threshold : float
        for IQR Default is 1.5,extreme 3. for mean_std is 3.

    Returns
    -------
    df : df
        pandas Dataframe without extreme outlier 

    """

    if method == 'mean_std':
        upper_fence = data.mean()+threshold*data.std()
        lower_fence = data.mean()-threshold*data.std()
    elif method == 'IQR':
        qt75 = data.quantile(0.75)
        qt25 = data.quantile(0.25)
        IQR = qt75-qt25
        upper_fence = qt75+threshold*IQR
        lower_fence = qt25-threshold*IQR

    df = data[(data > lower_fence) & (data < upper_fence)]
    return df
