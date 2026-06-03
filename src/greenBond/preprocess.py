
import re
import pandas as pd



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
    
