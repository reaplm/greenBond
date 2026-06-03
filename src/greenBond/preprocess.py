
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_target(df):
    #Cleaning The Data
    # For ProjectImplementationProgress do the following:
    # Captitalize (e.g Under construction -> Under Construction)
    # Remove Spaces (Under Construction -> UnderConstruction)
    # Standardize inconsiistent values 'In Operation' == 'Officially in operation' 
    # Replace NULL with Unknown
    df.loc[:, 'ProjectImplementationProgress'] =\
            (df['ProjectImplementationProgress']
              .str.title()
              .str.replace(' ', '', regex=False)
              .str.replace('OfficiallyInOperation', 'InOperation', regex=False)
              .fillna('Unknown')) 
    
    return df
    
def filter_co2(df):  
    # Filtering Carbon Emissions Data
    return df[df['IndicatorName'].str.\
                      contains('CO2|carbon|Carbon|CO₂', na=False, case=False)]


def encode_categorical(df):
    numeric_columns = \
        [
             'IssueSize', 
              'CouponRate',
              'IssueTerm', 
              'NumberOfProjectsInvested', 
              'TotalProjectInvestment',
              'AmountUsedForProject', 
              'ProjectEnvironmentalBenefits', 
              'EnvironmentalBenefitsPerUnitCapital',
              'BondEnvironmentalBenefits'
           ]
    
    text_columns = \
        [
            'BondClassification', 
            'BenefitInfoScope',
            'IndicatorName', 
            'NatureOfBenefitInformation',
         ]
        
    
    # --------------------------1. Remove Spaces & Punctuation---------------------
    for col in text_columns:
        df.loc[:, col] = df.loc[:, col]\
        .apply(lambda x: re.sub(r"[^\w]", "", x))
     
    # --------------------------2. Encode Target-----------------------------------
    # Create encoder
    target_encoder = LabelEncoder()
    
    # Encode the target
    target_encoded = target_encoder.fit_transform(df['ProjectImplementationProgress'])
    
    # --------------------------3. Encode Feature Columns----------------------------
    feature_df = pd.DataFrame()
    for col in text_columns:
        # Encoding ImplementationSchedule
        col_encoded = pd.get_dummies(df[col], prefix=col, dummy_na=False)
    
        # Convert from boolean (True/False) to integers (1/0) if preferred
        col_encoded = col_encoded.astype(int)
    
        # Add col_encoded to feature_df
        feature_df = pd.concat([feature_df, col_encoded], axis=1)
    
    # --------------------------4. Concatenate------------------------------------                                             
    # Add col_encoded to feature_df
    for col in numeric_columns:
        feature_df = pd.concat([feature_df, df[col]], axis=1)
        
    return feature_df, target_encoded