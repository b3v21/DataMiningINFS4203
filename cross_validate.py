import pandas as pd

from imputise import class_specifc, all_value

def cross_validate(file, k, imp_method):
    df = pd.read_csv(file)
    
    partition_percentage = 1/k
    samples = []
    
    for _ in range(k):
        partition = df.sample(frac = partition_percentage, ignore_index=False)
        df = df.drop(partition.index)
        samples.append(partition)
    
    for i in range(k):
        # imputise data using all samples except the ith sample as training data
        imputised_data = imp_method(pd.concat(samples[:i] + samples[i+1:]))
        imputised_data.to_csv('output/training_cv_{}_{}.csv'.format(imp_method.__name__,i), index=True)
        samples[i].to_csv('output/testing_cv_{}_{}.csv'.format(imp_method.__name__,i), index=True)
            

if __name__ == '__main__':
    cross_validate('data/train.csv', 5, class_specifc)