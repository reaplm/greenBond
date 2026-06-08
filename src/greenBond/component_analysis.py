import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def split_data(X, y):
    # Split the sample into 70% training and 30% test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size  = 0.3, 
        stratify=y,
        random_state=0)
    
    return X_train, X_test, y_train, y_test

    """
    The StandardScaler transforms the data so that the sample has a mean of 0 and 
    standard deviation of 1. This step is crucial for preventing variables with 
    large magnitudes such as BondEnvironmentalBenefits, could dominate the model 
    and prevent the estimator from learning from other variables.
    """
def scale_xdata(X_train, X_test):
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_scaled = sc.transform(X_train)
    X_test_scaled = sc.transform(X_test)

    return X_train_scaled, X_test_scaled

"""
Eigendecomposition was performed on the covariance matrix of the scaled training 
set to produce a set of eigenvalues. These values represent each principal 
component’s contribution to the total covariance.
"""
def calculate_covariance_matrix(X_train_scaled):
    # Covariance Matrix
    cov_mat = np.cov(X_train_scaled.T)
    eigen_vals, eigen_vecs = np.linalg.eigh(cov_mat)
    # print('\nEigenvalues \n%s' % eigen_vals)

    # Sort Eigenvalues
    total = sum(eigen_vals)
    var_exp = [(i/total) for i in sorted(eigen_vals, reverse = True)]
    cum_var_exp = np.cumsum(var_exp)
    
    return eigen_vals, eigen_vecs, var_exp, cum_var_exp

def plot_pc(var_exp, cum_var_exp):
    plt.figure(1)
    # Plot Eigenvalues (PCI vs Variance)
    x_range = range(1, len(var_exp) + 1)
    plt.bar(x_range, var_exp, alpha = 0.5, align = 'center',
            label = 'individual explained variance')
    plt.step(x_range, cum_var_exp, where = 'mid',
              label = 'cummulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc = 'best')
    plt.tight_layout()
    plt.savefig('Classification_features.png', dpi = 300)
    plt.show()

"""
The two most important principal components were selected to construct the 
projection matrix W. The matrix was then used to transform the training data 
X_train into PCA scores
"""
def plot_pc1_vs_pc2(X_train_scaled, X_test_scaled, y_train, 
                    eigen_vals, eigen_vecs, target_encoder):
    
    #List of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(eigen_vals))]    
    
    #Sort the (eigenvalue, eigenvector) tuples from high to low
    #Note: lambda is an anonymous callable function; k is parameter of function
    eigen_pairs.sort(reverse=True)
    w = np.hstack((eigen_pairs[0][1][:, np.newaxis],
                   eigen_pairs[1][1][:, np.newaxis]))
    print('Matrix W:\n', w)
    
    X_train_pca = X_train_scaled.dot(w)
    X_test_pca = X_test_scaled.dot(w)
    colors = ['r', 'b', 'g']
    markers = ['s', 'x', 'o']
    
    # Use class names in the legend
    class_names = target_encoder.classes_
    
    
    
    # =========================================================================
    #                 PLOTTING PCA
    # =========================================================================
        # 1. Create a figure canvas with 1 row and 2 columns
    # figsize=(width, height) in inches
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5),  sharex=True, sharey=True)

    # plt.figure(2)
    for l, c, m in zip(np.unique(y_train), colors, markers):
        class_name = class_names[l]
        ax1.scatter(X_train_pca[y_train == l, 0], 
                    X_train_pca[y_train == l, 1], 
                    c=c, label=class_name, marker=m)
            
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.legend(loc='lower left')
    ax1.set_title('PC1 vs PC2')
    # ax1.fig.savefig('green_bond_pca.png', dpi=300, bbox_inches='tight')
    
    # =========================================================================
    #                 PLOTTING OUTLIERS
    # =========================================================================
    # Find distance from the cluster to center (mean)
    center = np.mean(X_train_pca, axis=0)
    distances = np.linalg.norm(X_train_pca - center, axis=1)

    # Define outlier threshold (e.g., 95th percentile)
    threshold = np.percentile(distances, 95)
    outlier_mask = distances > threshold


    for l, c, m in zip(np.unique(y_train), colors, markers):
        class_name = class_names[l]
        ax2.scatter(X_train_pca[y_train == l, 0], 
                X_train_pca[y_train == l, 1], 
                c=c, label=class_name, marker=m)
    
    # Highlight outliers
    # Extracts all coordinates matching your outlier mask at once
    outlier_coords = X_train_pca[outlier_mask]
    
    ax2.scatter(outlier_coords[:, 0], outlier_coords[:, 1], 
                facecolors='none', 
                edgecolors='magenta', 
                s=100, 
                linewidths=2, 
                label='Outliers', 
                zorder=3) # Ensures the circles sit cleanly on top of data points
                
    # Synchronize Axis Constraints Explicitly
    x_min, x_max = -11, 1.5
    y_min, y_max = -5.4, 4.1
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    
    ax2.legend(loc='lower left') 
    ax2.set_title('Outliers')
    fig.savefig('pca_outliers.png', dpi=300, bbox_inches='tight')
    
    # Clean up overlaps and render
    plt.tight_layout()
    plt.show()

def print_loadings(X):
    # Display the loadings
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(X)
    
    pca_model = PCA(n_components=2)
    X_pca = pca_model.fit_transform(X)
    
    loadings = pd.DataFrame(
        pca_model.components_.T, 
        columns=['PC1', 'PC2'], 
        index=X.columns
    )
    
    # Set threshold (e.g., 0.05 for 5% significance)
    threshold = 0.05
    
    # Filter significant loadings (abs value > threshold for either PC)
    significant_vars = loadings[
        (abs(loadings['PC1']) > threshold) | 
        (abs(loadings['PC2']) > threshold)
    ].copy()
    
    print(f"Significant variables (threshold={threshold}):")
    print(significant_vars)


