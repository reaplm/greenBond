import pandas as pd
from pathlib import Path

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

if __name__ == "__main__":
    main()
