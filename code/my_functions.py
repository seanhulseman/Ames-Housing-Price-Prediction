import pandas as pd
import numpy as np
def map_home(df): #Note - Dictionary for maps are contraversial in my mind. Subject to change values
    df.columns =df.columns.str.lower().str.replace(' ','_')# makes things easier for me
    # fence column was split into two ad the original was dropped 
    df = df.drop(columns=['id']) # id is arbitrary, lot shape was hard to deal with last time and I will leave it out this time too
    df['fence_private']= df['fence'].map({'GdPrv':2,'MnPrv':1,'GdWo':0,'MnWw':0,np.nan:0})
    df['fence_wood'] = df['fence'].map({'GdPrv':0,'MnPrv':0,'GdWo':2,'MnWw':1,np.nan:0})
    df.drop(columns=['fence'], inplace=True)
    # this function is based off of wild assumptions by me. The value of the ordinal variables here relative to eachother is an educated guess
    qual_map = {'Po':1,'Fa':2,'TA':3,'Ex':5,'Gd':4,np.nan:0}
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
    houses_dummy_1= pd.get_dummies(data=df,columns=['mas_vnr_type','bsmt_exposure','garage_type','alley','misc_feature','land_contour'], drop_first=True,dummy_na=True)
    # this group of columns has nan values that I am not comfortable calling 0

    houses_dummy_2= pd.get_dummies(data=houses_dummy_1,columns=['ms_subclass','ms_zoning','street','neighborhood','condition_2','bldg_type',
                                                            'house_style','roof_style','roof_matl','exterior_1st','exterior_2nd','foundation',
                                                           'central_air','electrical','sale_type','condition_1','utilities','lot_config','land_slope',
                                                           'heating','paved_drive'], drop_first=True)
    return houses_dummy_2.copy()



def multiplier_attempt3(houses3):
    # lot_area stays but now make lot_area_value which depends on: lot_shape 
    houses3['lot_area_frontage']=houses3['remainder__lot_area']*houses3['remainder__lot_frontage']


    # garages: Area: 'garage_finish',garage_qual,garage_cond, 
    houses3['garage_area_qual']=houses3['remainder__garage_area']*houses3['remainder__garage_qual']
    houses3['garage_area_cond']=houses3['remainder__garage_area']*houses3['remainder__garage_cond']
    houses3['garage_area_finish']=houses3['remainder__garage_area']*houses3['remainder__garage_finish']

    # fireplaces: fireplace_qu
    
    # bsmtfin_sf_1 : bsmtfin_type_1, 
    houses3['bsmt_sf_qual_type_1']=houses3['remainder__bsmtfin_sf_1']*houses3['remainder__bsmtfin_type_1']
    # bsmtfin_sf_2: bsmtfin_type_2, 
    houses3['bsmt_total_sf_cond']=houses3['remainder__total_bsmt_sf']*houses3['remainder__bsmt_cond']
    #total_bsmt_sf: bsmt_qual, bsmt_cond 
    houses3['bsmt_total_sf_qual']=houses3['remainder__total_bsmt_sf']*houses3['remainder__bsmt_qual']   
    
    #houses3['misc_feature_Gar2_val']=houses3['oh__misc_feature_Gar2']*houses3['remainder__misc_val']
    
    houses3['misc_feature_Othr_val']=houses3['oh__misc_feature_Othr']*houses3['remainder__misc_val']
    houses3['misc_feature_Shed_val']=houses3['oh__misc_feature_Shed']*houses3['remainder__misc_val']
    
    houses3['misc_feature_TenC_val']=houses3['oh__misc_feature_TenC']*houses3['remainder__misc_val']
    
    # testing out different variations of dropping codependent columns
    houses3.drop(columns=['remainder__garage_qual','remainder__garage_cond','remainder__garage_finish','oh__garage_type_Attchd','oh__garage_type_Basment','oh__garage_type_BuiltIn','oh__garage_type_CarPort','oh__garage_type_Detchd','remainder__fireplace_qu','remainder__exter_cond','remainder__bsmt_qual','remainder__bsmt_cond','remainder__pool_qc','remainder__pool_area'])
    return houses3.copy()