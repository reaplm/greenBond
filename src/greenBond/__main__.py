import pandas as pd
from pathlib import Path
from preprocess import *

def load_data():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    path = BASE_DIR / "data" / "green_n_transition_v1_clean.csv"

    try:
        df = pd.read_csv(path)
        print("Successfully loaded data....")
        return df
    except FileNotFoundError:
        print("Failed to load file. Please check location.")
    
    

def main():
    print("Starting Green Bond Analysis.....")
    
    # Loading the data
    green_benefit_df = load_data()
    
    # Print dataframe info
    print(green_benefit_df.info())
    print('---------------------Rows and Columns------------------------------')
    print(f'Rows:{ green_benefit_df.shape[0]} x Columns:{green_benefit_df.shape[1]}\n')
    
    print('---------------------Target Class --------------------------------\n')
    class_counts = green_benefit_df['ProjectImplementationProgress'].value_counts()
    print(class_counts)
    
    # Clean target variable
    clean_df = clean_target(green_benefit_df)
    
    # Filter CO2
    co2_df = filter_co2(clean_df)
    
    # print(co2_df['ProjectImplementationProgress'].value_counts())
    
    # Encode categorical
    X, y = encode_categorical(co2_df)

    

if __name__ == "__main__":
    main()
