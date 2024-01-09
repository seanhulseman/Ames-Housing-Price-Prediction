import pandas as pd
import numpy as np
def map_home(df): #Note - Dictionary for maps are contraversial in my mind. Subject to change values
    df.columns =df.columns.str.lower().str.replace(' ','_')# makes things easier for me
    # fence column was split into two ad the original was dropped 
    df = df.drop(columns=['id']) # id is arbitrary, lot shape was hard to deal with last time and I will leave it out this time too
    # fences had data on the quality of the fence and the type of fence. I will split these into two columns
    df['fence_private']= df['fence'].map({'GdPrv':2,'MnPrv':1,'GdWo':0,'MnWw':0,np.nan:0})
    df['fence_wood'] = df['fence'].map({'GdPrv':0,'MnPrv':0,'GdWo':2,'MnWw':1,np.nan:0})
    df.drop(columns=['fence'], inplace=True)
    # this function is based off of wild assumptions by me. The value of the ordinal variables here relative to eachother is an educated guess
    # I will be mapping the ordinal variables to number. The question is this: 
    # What is the most affective way to map these qualities to numbers so they reflect the reality of property. Property value increases for many reasons but one main consideration us that poor quality property is a financial burden.
    # my original qual_map = {'Po':1,'Fa':2,'TA':3,'Ex':4,'Gd':5,np.nan:0}
    qual_map = {'Po':.25,'Fa':.5,'TA':1,'Ex':1.5,'Gd':2,np.nan:0} # trying this out now
    
    df['garage_finish']= df['garage_finish'].map({'Unf':1,'RFn':2,'Fin':3,np.nan:0})
    df['garage_qual']= df['garage_qual'].map(qual_map)
    df['garage_cond']= df['garage_cond'].map(qual_map)
    df['bsmtfin_type_2']= df['bsmtfin_type_2'].map({'Unf':.5,'Rec':1.5,'LwQ':1,'BLQ':2,'ALQ':2.5,'GLQ':3,np.nan:0}) # i have decided to make these ordinal!
    df['bsmtfin_type_1']= df['bsmtfin_type_1'].map({'Unf':.5,'Rec':1.5,'LwQ':1,'BLQ':2,'ALQ':2.5,'GLQ':3,np.nan:0}) # i have decided to make these ordinal!
    df['fireplace_qu']=df['fireplace_qu'].map(qual_map)
    df['bsmt_qual']=df['bsmt_qual'].map(qual_map)
    df['bsmt_cond']=df['bsmt_cond'].map(qual_map)
    df['pool_qc']=df['pool_qc'].map(qual_map) 
    df['exter_qual']= df['exter_qual'].map(qual_map)
    df['exter_cond']= df['exter_cond'].map(qual_map)
    df['heating_qc']= df['heating_qc'].map(qual_map)
    df['kitchen_qual']= df['kitchen_qual'].map(qual_map)
    df['functional']=df['functional'].map({'Typ':8,'Min1':7,'Min2':6,'Mod':5,'Maj1':4,'Maj2':3,'Sev':2,'Sal':1}) # this could be catagorical but i will try to ordinalize it here
    return df.copy()
def numeric_houses(df):
    df['mas_vnr_area'] = np.where(df['mas_vnr_area'].isna(),0,df['mas_vnr_area'].astype(float)) 
    df['bsmt_full_bath']=np.where(df['bsmt_full_bath'].isna(),0,df['bsmt_full_bath'].astype(float)) 
    df['bsmt_half_bath']=np.where(df['bsmt_half_bath'].isna(),0,df['bsmt_half_bath'].astype(float)) 
    df['bsmtfin_sf_1']=np.where(df['bsmtfin_sf_1'].isna(),0,df['bsmtfin_sf_1'].astype(float)) 
    df['bsmtfin_sf_2']=np.where(df['bsmtfin_sf_2'].isna(),0,df['bsmtfin_sf_2'].astype(float)) 
    df['bsmt_unf_sf']=np.where(df['bsmt_unf_sf'].isna(),0,df['bsmt_unf_sf'].astype(float)) 
    df['total_bsmt_sf']=np.where(df['total_bsmt_sf'].isna(),0,df['total_bsmt_sf'].astype(float)) 
    df['lot_frontage'] = np.where(df['lot_frontage'].isna(),0,df['lot_frontage'].astype(float)) 
    df['garage_area'] = np.where(df['garage_area'].isna(),0,df['garage_area'].astype(float)) 
    df['garage_yr_blt'] = np.where(df['garage_yr_blt'].isna(),0,df['garage_yr_blt'].astype(float))  
    df['garage_cars']=np.where(df['garage_cars'].isna(),0,df['garage_cars'].astype(float)) 
    return df.copy()
def dummy_houses(df): #not used anymore
    houses_dummy_1= pd.get_dummies(data=df,columns=['mas_vnr_type','bsmt_exposure','garage_type','alley','misc_feature','land_contour','lot_shape'], drop_first=True,dummy_na=True)
    # this group of columns has nan values that I am not comfortable calling 0
    houses_dummy_2= pd.get_dummies(data=houses_dummy_1,columns=['ms_subclass','ms_zoning','street','neighborhood','condition_2','bldg_type',
                                                            'house_style','roof_style','roof_matl','exterior_1st','exterior_2nd','foundation',
                                                           'central_air','electrical','sale_type','condition_1','utilities','lot_config','land_slope',
                                                           'heating','paved_drive'], drop_first=True)
    return houses_dummy_2.copy()


def quality_multiplication(df):
    # lot_area stays but now make lot_area_value which depends on: lot_shape 
    df['lot_area_frontage']=df['lot_area']*df['lot_frontage']

    # I want to find all the the specific dummy variable columns that can be correlated with some measure of quality or condition
    # I will then use these columns to create a new feature that is the multiplication of the dummy variable and the quality or condition
    # this will give me a new feature that is a measure of the quality or condition of the house feature
    # I will then use this new feature to see if it improves the model
    # exterior multiplier
    desired_columns_ext = [col for col in list(df.columns) if 'exterior_1st' in col or 'exterior_2nd' in col or 'foundation' in col] # 'exterqual', 'extercond' for the first 4. 'roofstyle' times 'roofmatl' to be comprehensive 
    for col in desired_columns_ext:
        df[col+'_qual']=df[col]*df['exter_qual']
        df[col+'_cond']=df[col]*df['exter_cond']
        df.drop(columns=[col],inplace=True)
    roof_mats = [col for col in list(df.columns) if 'roof_matl' in col] 
    roof_styles = [col for col in list(df.columns) if 'roof_style' in col]
    for mat in roof_mats: 
        df[mat+'qual']=df[mat]*df['exter_qual']
        df[mat+'cond']=df[mat]*df['exter_cond']

    for style in roof_styles: 
        df[style+'qual']=df[style]*df['exter_qual']
        df[style+'cond']=df[style]*df['exter_cond']
    # desired_columns_heating = [col for col in list(df.columns) if 'heating' in col and col!='heatingqc'] # 'heatingqc'
    # if 'heatingqc' in desired_columns_heating:
    #     desired_columns_heating
    # for col in desired_columns_heating:
    #     df[col+'_qual']=df[col]*df['heatingqc']
    #     df.drop(columns=[col],inplace=True)
    desired_columns_garage = [col for col in list(df.columns) if 'garage_type' in col] # 'garagequal', 'garagecond', 'garagearea' like below
    for col in desired_columns_garage:
        df[col+'_area'+'_qual']=df[col]*df['garage_area']*df['garage_qual']
        df[col+'_area'+'_cond']=df[col]*df['garage_area']*df['garage_cond']
        df.drop(columns=[col],inplace=True)
    desired_columns_misc = [col for col in list(df.columns) if 'misc_feature' in col] # 'miscval'
    for col in desired_columns_misc:
        df[col+'_val']=df[col]*df['misc_val']
        df.drop(columns=[col],inplace=True)
    # finishing up garage detail
    # df['garage_finish_qual']=df['garage_finish']*df['garage_qual']
    # df.drop(columns=['garage_finish'],inplace=True)
    # df['garage_finish_cond']=df['garage_finish']*df['garage_cond']
    # df.drop(columns=['garage_cond'],inplace=True)
    # df.drop(columns=['garage_qual'],inplace=True)

    # fireplaces: fireplace_qu
    if 'fireplace_qu' in list(df.columns):
        df['fireplace_qu_val']=df['fireplace_qu']*df['fireplaces']
    else:
        df['fireplace_qu_val']=0
    df.drop(columns=['fireplace_qu'],inplace=True)

    # pool - pool quality, pool area
    if 'pool_qc' in list(df.columns):
        df['pool_qc_area']=df['pool_qc']*df['pool_area']
    else:
        df['pool_qc_area']=0
    df.drop(columns=['pool_qc'],inplace=True)
    df.drop(columns=['pool_area'],inplace=True)

    # basement - bsmt_qual, bsmt_cond, bsmtfin_sf_1, bsmtfin_sf_2, bsmt_unf_sf, total_bsmt_sf
    df['bsmt_qual_area'] = df['bsmt_qual']*df['total_bsmt_sf']
    df['bsmt_cond_area'] = df['bsmt_cond']*df['total_bsmt_sf']
    df.drop(columns=['bsmt_qual','bsmt_cond'],inplace=True)


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
