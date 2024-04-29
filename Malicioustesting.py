import pandas as pd
import numpy as np
from featurePrepare import getUrlFeatures
from ml_loader import ctree_predict, RF_predict, SVM_predict, c45_predict  

def main():
    # Load URLs from CSV without headers
    df = pd.read_csv("Malicious.csv", header=None)
    
    # Optional: Name the first column for easier access
    df.columns = ['URL']
    
    # Prepare for output
    results = {'Ctree': [], 'RF': [], 'SVM': [], 'C4.5': []} 
    
    # Process each URL
    for index, row in df.iterrows():
        url = row['URL']
        features = getUrlFeatures(url)
        
        if features is not None:
            # Make predictions
            ctree_pred = np.clip(ctree_predict(features), 0, 1) * 100
            rf_pred = np.clip(RF_predict(features), 0.1, 1) * 100
            svm_pred = np.clip(SVM_predict(features), 0.3, 0.9) * 100
            c45_pred = np.clip(c45_predict(features), 0, 1) * 100  
            
            # Store results
            results['Ctree'].append(f"{ctree_pred[0]:.2f}%")
            results['RF'].append(f"{rf_pred[0]:.2f}%")
            results['SVM'].append(f"{svm_pred[0]:.2f}%")
            results['C4.5'].append(f"{c45_pred[0]:.2f}%") 
        else:
            # Handle cases where features could not be prepared
            results['Ctree'].append("N/A")
            results['RF'].append("N/A")
            results['SVM'].append("N/A")
            results['C4.5'].append("N/A")  
    
    # Create DataFrame for results
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    results_df.to_csv(r'C:\Users\sauls\source\repos\COMP3000HK23_24_SzeChiKeung\Results.csv', index=False)

if __name__ == "__main__":
    main()
