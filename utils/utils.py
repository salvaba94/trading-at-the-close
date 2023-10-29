import pandas as pd

def merge_df(df1, df2, on):

    if len(df1) == 0:
        return df2
    elif len(df2) == 0:
        return df1
    else:
        df3 = pd.merge(df1, df2, on=on, how='outer')
        #now we have a mess to fix
        cols=[x[:-2] for x in df3.columns if x.endswith('_x')]
        for i_col in cols:
            df3.loc[:,i_col+'_x']=df3[i_col+'_x'].combine_first(df3[i_col+'_y'])
            df3.rename(columns={i_col+'_x':i_col},inplace=True)
            df3.drop(columns=[i_col+'_y'],inplace=True)
        return df3