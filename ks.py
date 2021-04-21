# TO-DO
# 1. Add inverse CDF (1 - CDF)
# 2. Change dataframe's column names
def ks(score, target, bins=10, negative_relationship=True, round_intervals=False, print_table=False, write_table=False):
    '''
    Prints the Kolmogorov-Smirnov metric and returns a summary table.

            Parameters
            ----------
                    score : array_like
                        Input array or object that can be converted to an array. Each element in the array
                        represents the likelihood of an event. Must be the same length as `target`.
                    target : array_like
                        Input array or object that can be converted to an array. Each element in the array
                        must be 1 if the event was observed or 0 if the event was not observed. Must be the same
                        length as `score`.
                    bins : int, default `10`
                        Integer that indicates the number of equally-sized bins to split the data into.
                    negative_relationship : bool, default `True`
                        Determine the relationship between the score and the target.
                        If `negative_relationship == True`, then a LOW score means HIGH probability of default.
                        If `negative_relationship == False`, then a LOW score means LOW probability of default.
                    round_intervals : bool, default `False`
                        Determine whether or not to round and convert the bin edges to integers.
                    print_table : bool, default `False`
                        Determine whether or not to print the summary table.
                    write_table : bool, default `False`
                        Determine whether or not to write the summary table as a .csv file.
                        If `write_table == True`, the file will be written as a .csv file in the current working directory.
            
            Returns
            ----------
                    Two-dimensional summary table stored in a pandas.DataFrame object.
            
            Notes
            ----------
                    Switching `negative_relationship` from `True` to `False` will slightly alter the KS metric.
                    The reason behind this change is the way the cutoff is implemented. The difference is negligible.
    '''
    import numpy as np
    import pandas as pd
    temp = pd.DataFrame({'score':score, 'target':target})
    temp['bin'] = pd.qcut(temp['score'], bins, duplicates='drop')
    if round_intervals:
        temp['bin'] = temp['bin'].apply(lambda x: pd.Interval(int(round(x.left, 0)), int(round(x.right, 0))))
    temp = temp.groupby('bin').agg({'target':['size','sum','mean']}).reset_index()
    temp.columns = ['bin','count','bads','bad_rate_in_bin']
    temp['goods'] = temp['count'] - temp['bads']
    if negative_relationship == False:
        temp = temp.sort_index(ascending=False)
    temp['cumulative_bad_rate'] = temp['bads'].cumsum().div(temp['bads'].sum())
    temp['cumulative_good_rate'] = temp['goods'].cumsum().div(temp['goods'].sum())
    temp['ks'] = np.abs(temp['cumulative_bad_rate'] - temp['cumulative_good_rate'])
    temp['remaining_bad_rate'] = (temp['bads'].sum() - temp['bads'].cumsum().shift(1, fill_value=0)) \
                                 .div(temp['count'].sum() - temp['count'].cumsum().shift(1, fill_value=0))
    if print_table:
        print(temp, '\n\n')
    if write_table:
        temp.to_csv('ks_summary_table.csv', index=False)
    print('KS:', round(temp['ks'].max(), 2), 'out of 1.00')
    cols = ['bin','count','bads','goods','bad_rate_in_bin','cumulative_bad_rate','cumulative_good_rate',
            'ks','remaining_bad_rate']
    return temp[cols].sort_index()
