import pandas as pd

from imputise import class_specifc, all_value
from normalisation import standardise, min_max_normalise
from clean import gaussian_outlier_detection
from classifiers import k_NN
from metrics import manhattan, mc_f1

def cross_validate(file, k, imp_method, norm_method, clean_method, classifier):
    df = pd.read_csv(file)
    
    partition_percentage = 1/k
    samples = []
    
    for _ in range(k):
        partition = df.sample(frac = partition_percentage, ignore_index=False)
        df = df.drop(partition.index)
        samples.append(partition)
    
    
    mc_f1_scores = []
    for i in range(k):
        print(f"Calculating fold {i+1} of {k}")
        
        # apply preprocessing to train data
        imputised_train_data = imp_method(pd.concat(samples[:i] + samples[i+1:]))
        cleaned_train_data = clean_method(imputised_train_data)
        normalised_train_data = norm_method(cleaned_train_data, cols = range(0,100))

        # apply preprocessing to test data
        imputised_test_data = imp_method(samples[i])
        cleaned_test_data = clean_method(imputised_test_data)
        normalised_test_data = norm_method(cleaned_test_data, cols = range(0,100))
        
        predicted_labels = k_NN(5, normalised_train_data, normalised_test_data, norm_method, manhattan)
        true_labels = normalised_test_data['Label'].to_dict()
        
        mc_f1_res = mc_f1(list(true_labels.values()), list(predicted_labels.values()))
        print(mc_f1_res)
        
        mc_f1_scores.append(mc_f1_res)
        
    return sum(mc_f1_scores)/k
            

if __name__ == '__main__':
    res = cross_validate('data/train.csv', 5, class_specifc, standardise, min_max_normalise, k_NN)
    print("AVERAGE MACRO f1:", round(res,2))
    