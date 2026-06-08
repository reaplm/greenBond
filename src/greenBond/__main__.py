import pandas as pd
from pathlib import Path
from preprocess import clean_target, filter_co2, encode_categorical
from component_analysis import split_data,scale_xdata, calculate_covariance_matrix, plot_pc, plot_pc1_vs_pc2
from random_forest_ml import TrainRandomForest

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
    X, y, class_names = encode_categorical(co2_df)
  
    # PCA
    # Split the data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Scale
    X_train_scaled, X_test_scaled = scale_xdata(X_train, X_test)
    
    # Eigendecomposition
    eigen_vals, eigen_vecs, var_exp, cum_var_exp = calculate_covariance_matrix(X_train_scaled)
    
    plot_pc(var_exp, cum_var_exp)
    plot_pc1_vs_pc2(X_train_scaled, X_test_scaled, y_train, 
                    eigen_vals, eigen_vecs, class_names)

    # =========================================================================
    #                 RANDOM FOREST
    # =========================================================================
    random_forest = TrainRandomForest( X_train_scaled, X_test_scaled, y_train, y_test)
    random_forest.train_random_forest(eigen_vals, eigen_vecs)
    random_forest.display_confusion_matrix()
    
if __name__ == "__main__":
    main()
