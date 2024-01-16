import pandas as pd
import numpy as np
def map_home(df_original): 
    # column preprocessing
    df = df_original.copy()
    df.columns =df.columns.str.lower().str.replace(' ','').str.replace('/','')
    if 'pid' in list(df.columns):
        df.drop(columns=['pid'],inplace=True)
    if 'id' in list(df.columns):
        df.drop(columns=['id'],inplace=True)
    # fence column has 2 scales of quality. I will make 2 columns for each fence type. One for wood and one for privacy
    df['fenceprivate']= df['fence'].map({'GdPrv':2,'MnPrv':1,'GdWo':0,'MnWw':0,np.nan:0})
    df['fencewood'] = df['fence'].map({'GdPrv':0,'MnPrv':0,'GdWo':2,'MnWw':1,np.nan:0})
    df.drop(columns=['fence'], inplace=True)
    # this function is based off of assumptions by me that are based on hunches and may be useful to vary. The value of the ordinal variables here relative to eachother is an educated guess
    # I will be mapping the ordinal variables to a scale. The question is this: 
    # What is the most affective way to map these qualities to numbers so they reflect the reality of property. Property value increases for many reasons but one main consideration us that poor quality property is a financial burden.
    # my original qual_map = {'Po':1,'Fa':2,'TA':3,'Ex':4,'Gd':5,np.nan:0}
    #{'Po':.25,'Fa':.5,'TA':1,'Ex':4,'Gd':2,np.nan:0} # would this be better?
    # {'Po':.25,'Fa':.5,'TA':1,'Ex':2,'Gd':1.5,np.nan:0} # trying this out now
    qual_map = {'Po':.25,'Fa':.5,'TA':1,'Ex':4,'Gd':2,np.nan:0} # trying this out now
    
    df['garagefinish']= df['garagefinish'].map({'Unf':1,'RFn':2,'Fin':3,np.nan:0})
    df['garagequal']= df['garagequal'].map(qual_map)
    df['garagecond']= df['garagecond'].map(qual_map)
    df['bsmtfintype2']= df['bsmtfintype2'].map({'Unf':.5,'Rec':1.5,'LwQ':1,'BLQ':2,'ALQ':2.5,'GLQ':3,np.nan:0}) # i have decided to make these ordinal!
    df['bsmtfintype1']= df['bsmtfintype1'].map({'Unf':.5,'Rec':1.5,'LwQ':1,'BLQ':2,'ALQ':2.5,'GLQ':3,np.nan:0}) # i have decided to make these ordinal!
    df['fireplacequ']=df['fireplacequ'].map(qual_map)
    df['bsmtqual']=df['bsmtqual'].map(qual_map)
    df['bsmtcond']=df['bsmtcond'].map(qual_map)
    df['poolqc']=df['poolqc'].map(qual_map) 
    df['exterqual']= df['exterqual'].map(qual_map)
    df['extercond']= df['extercond'].map(qual_map)
    df['heatingqc']= df['heatingqc'].map(qual_map)
    df['kitchenqual']= df['kitchenqual'].map(qual_map)
    df['functional']=df['functional'].map({'Typ':1,'Min1':7/8,'Min2':6/8,'Mod':5/8,'Maj1':4/8,'Maj2':3/8,'Sev':2/8,'Sal':1/8, np.nan:0}) # this could be catagorical but i will try to ordinalize it here
    return df.copy()
def numeric_houses(df):
    df['masvnrarea'] = np.where(df['masvnrarea'].isna(),0,df['masvnrarea'].astype(float)) 
    df['bsmtfullbath']=np.where(df['bsmtfullbath'].isna(),0,df['bsmtfullbath'].astype(float)) 
    df['bsmthalfbath']=np.where(df['bsmthalfbath'].isna(),0,df['bsmthalfbath'].astype(float)) 
    df['bsmtfinsf1']=np.where(df['bsmtfinsf1'].isna(),0,df['bsmtfinsf1'].astype(float)) 
    df['bsmtfinsf2']=np.where(df['bsmtfinsf2'].isna(),0,df['bsmtfinsf2'].astype(float)) 
    df['bsmtunfsf']=np.where(df['bsmtunfsf'].isna(),0,df['bsmtunfsf'].astype(float)) 
    df['totalbsmtsf']=np.where(df['totalbsmtsf'].isna(),0,df['totalbsmtsf'].astype(float)) 
    df['lotfrontage'] = np.where(df['lotfrontage'].isna(),0,df['lotfrontage'].astype(float)) 
    df['garagearea'] = np.where(df['garagearea'].isna(),0,df['garagearea'].astype(float)) 
    df['garageyrblt'] = np.where(df['garageyrblt'].isna(),0,df['garageyrblt'].astype(float))  
    #df.loc[df['garageyrblt'] == 0, 'garageyrblt'] = df['yearbuilt'] # I do not like this being 
    df['garagecars']=np.where(df['garagecars'].isna(),0,df['garagecars'].astype(float)) 
    # i want to make a columns th takes 0 from the yearremodadd column and replaces them with the yearbuilt using np.where
    df.loc[df['yearremodadd'] == 0, 'yearremodadd'] = df['yearbuilt']
    # replace 0 in the functional column with the mean 
    df['functional']=np.where(df['functional'].isna(),df['functional'].mean(),df['functional'])
    return df.copy()
def dummy_houses(df): #not used anymore
    houses_dummy_1= pd.get_dummies(data=df,columns=['masvnrtype','bsmtexposure','garagetype','alley','miscfeature','landcontour','lotshape'], drop_first=True,dummy_na=True)
    # this group of columns has nan values that I am not comfortable calling 0
    houses_dummy_2= pd.get_dummies(data=houses_dummy_1,columns=['mssubclass','mszoning','street','neighborhood','condition2','bldgtype',
                                                            'housestyle','roofstyle','roofmatl','exterior1st','exterior2nd','foundation',
                                                           'centralair','electrical','saletype','condition1','utilities','lotconfig','landslope',
                                                           'heating','paveddrive'], drop_first=True)
    return houses_dummy_2.copy()


def quality_multiplication(df):
    # I want to find all the the specific dummy variable columns that can be correlated with some measure of quality or condition
    # I will then use these columns to create a new feature that is the multiplication of the dummy variable and the quality or condition
    # I will then use this new feature to see if it improves the model
    # exterior multiplier
    desired_columns_ext = [col for col in list(df.columns) if 'exterior1st' in col or 'exterior2nd' in col or 'foundation' in col] # 'exterqual', 'extercond' for the first 4. 'roofstyle' times 'roofmatl' to be comprehensive 
    for col in desired_columns_ext:
        df[col+'_qual']=df[col]*df['exterqual']
        df[col+'_cond']=df[col]*df['extercond']
        df.drop(columns=[col],inplace=True)
    desired_columns_roof = [col for col in list(df.columns) if 'roofmatl' in col] 
    for col in desired_columns_roof:
        df[col+'_qual']=df[col]*df['overallqual']
        df[col+'_cond']=df[col]*df['overallcond']
        df.drop(columns=[col],inplace=True)
    desired_columns_heating = [col for col in list(df.columns) if 'heating' in col and col!='heatingqc'] # 'heatingqc'
    for col in desired_columns_heating:
        df[col+'_qual']=df[col]*df['heatingqc']
        df[col+'_cond']=df[col]*df['overallqual']
        df.drop(columns=[col],inplace=True)
    desired_columns_garage = [col for col in list(df.columns) if 'garage_type' in col] 
    for col in desired_columns_garage:
        df[col+'_area'+'_qual']=df[col]*df['garagearea']*df['garagequal']
        df[col+'_area'+'_cond']=df[col]*df['garagearea']*df['garagecond']
        df.drop(columns=[col],inplace=True)
    desired_columns_misc = [col for col in list(df.columns) if 'miscfeature' in col] # 'miscval'
    for col in desired_columns_misc:
        df[col+'_val']=df[col]*df['miscval']
        #df.drop(columns=[col],inplace=True)
    # quantify the neighborhood effect on certain types of housing
    desired_columns_neighborhood = [col for col in list(df.columns) if 'neighborhood' in col] # 'neighborhood'
    desired_columns_subclass = [col for col in list(df.columns) if 'mssubclass' in col] 
    for col in desired_columns_subclass:
        df[col+'_qual']=df[col]*df['overallqual']
        #df[col+'_cond']=df[col]*df['overall_cond']
        df.drop(columns=[col],inplace=True)
    for col in desired_columns_neighborhood:
        # I want to multiply the neighborhood by overall quality
        df[col+'_qual']=df[col]*df['overallqual']
        df[col+'_cond']=df[col]*df['overallcond']
        #df.drop(columns=[col],inplace=True)
     # year sold should be a decimal combination of the year and month. 
    df['year_sold']=df['yrsold']+df['mosold']/12
    df.drop(columns=['yrsold'],inplace=True)
    

    # fireplaces: fireplace_qu
    if 'fireplace_qu' in list(df.columns):
        df['fireplace_qu_val']=df['fireplacequ']*df['fireplaces']
    else:
        df['fireplace_qu_val']=0
    df.drop(columns=['fireplacequ'],inplace=True)

    # pool - pool quality, pool area
    if 'pool_qc' in list(df.columns):
        df['pool_qc_area']=df['poolqc']*df['poolarea']
    else:
        df['pool_qc_area']=0
    df.drop(columns=['poolqc'],inplace=True)
    df.drop(columns=['poolarea'],inplace=True)

 
 
    return df.copy()

# This is happening because of the function
def test_col_matcher(X_train,X_test):
    train_cols = list(X_train.columns)
    test_cols = list(X_test.columns)
    # I will be adding columns to the test set that are not in the training set and setting them to 0
    for col in train_cols:
        if col not in test_cols:
            X_test[col]=0
    # I will not be using information on ameneities that are not in the test set
    for col in test_cols:
        if col not in train_cols:
            X_test.drop(columns=[col],inplace=True)
    X_test = X_test[X_train.columns]
    return X_test.copy()
