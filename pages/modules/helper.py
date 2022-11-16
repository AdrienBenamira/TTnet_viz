import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from tqdm import tqdm


def read_info(info_path):
    with open(info_path) as f:
        f_list = []
        for line in f:
            tokens = line.strip().split()
            f_list.append(tokens)
    return f_list[:-1], int(f_list[-1][-1])


def read_csv(data_path, info_path, shuffle=False):
    D = pd.read_csv(data_path, header=None)
    if shuffle:
        D = D.sample(frac=1, random_state=0).reset_index(drop=True)
    f_list, label_pos = read_info(info_path)
    f_df = pd.DataFrame(f_list)
    D.columns = f_df.iloc[:, 0]
    y_df = D.iloc[:, [label_pos]]
    X_df = D.drop(D.columns[label_pos], axis=1)
    f_df = f_df.drop(f_df.index[label_pos])
    return X_df, y_df, f_df, label_pos


class DBEncoder:
    """Encoder used for data discretization and binarization."""

    def __init__(self, f_df, discrete=False, y_one_hot=True, drop='first'):
        self.f_df = f_df
        self.discrete = discrete
        self.y_one_hot = y_one_hot
        self.label_enc = preprocessing.OneHotEncoder(categories='auto') if y_one_hot else preprocessing.LabelEncoder()
        self.feature_enc = preprocessing.OneHotEncoder(categories='auto', drop=drop)
        self.imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.X_fname = None
        self.y_fname = None
        self.discrete_flen = None
        self.continuous_flen = None
        self.mean = None
        self.std = None

    def split_data(self, X_df):
        discrete_data = X_df[self.f_df.loc[self.f_df[1] == 'discrete', 0]]
        continuous_data = X_df[self.f_df.loc[self.f_df[1] == 'continuous', 0]]
        if not continuous_data.empty:
            continuous_data = continuous_data.replace(to_replace=r'.*\?.*', value=np.nan, regex=True)
            continuous_data = continuous_data.astype(np.float)
        return discrete_data, continuous_data

    def fit(self, X_df, y_df):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        self.label_enc.fit(y_df)
        self.y_fname = list(self.label_enc.get_feature_names(y_df.columns)) if self.y_one_hot else y_df.columns

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if do not discretize them.
            self.imp.fit(continuous_data.values)
        if not discrete_data.empty:
            # One-hot encoding
            self.feature_enc.fit(discrete_data)
            feature_names = discrete_data.columns
            self.X_fname = list(self.feature_enc.get_feature_names(feature_names))
            self.discrete_flen = len(self.X_fname)
            if not self.discrete:
                self.X_fname.extend(continuous_data.columns)
        else:
            self.X_fname = continuous_data.columns
            self.discrete_flen = 0
        self.continuous_flen = continuous_data.shape[1]

    def transform(self, X_df, y_df, normalized=False, keep_stat=False):
        X_df = X_df.reset_index(drop=True)
        y_df = y_df.reset_index(drop=True)
        discrete_data, continuous_data = self.split_data(X_df)
        # Encode string value to int index.
        y = self.label_enc.transform(y_df.values.reshape(-1, 1))
        if self.y_one_hot:
            y = y.toarray()

        if not continuous_data.empty:
            # Use mean as missing value for continuous columns if we do not discretize them.
            continuous_data = pd.DataFrame(self.imp.transform(continuous_data.values),
                                           columns=continuous_data.columns)
            if normalized:
                if keep_stat:
                    self.mean = continuous_data.mean()
                    self.std = continuous_data.std()
                continuous_data = (continuous_data - self.mean) / self.std
        if not discrete_data.empty:
            # One-hot encoding
            discrete_data = self.feature_enc.transform(discrete_data)
            if not self.discrete:
                X_df = pd.concat([pd.DataFrame(discrete_data.toarray()), continuous_data], axis=1)
            else:
                X_df = pd.DataFrame(discrete_data.toarray())
        else:
            X_df = continuous_data
        return X_df.values, y
    
    
def find_splits(db_enc):
    names = db_enc.X_fname[:db_enc.discrete_flen]
    temp = names[0][:2]
    idx = [0]
    for k in range(len(names)):
        if temp != names[k][:2]:
            idx.append(k)
            temp = names[k][:2]
            
    return idx
        

def sample_generator(data, db_enc, size, limit=94 ):
    #Split the dataset manually
    data_dc, data_c = data[:,:limit], data[:,limit:]
    idx = find_splits(db_enc)
    
    #===================Generating the discrete part=======================#   
    prob = np.sum(data_dc, axis = 0)/len(data_dc)
        
    splitted_prob = []
    for i in range(len(idx) - 1):
        splitted_prob.append(prob[idx[i] : idx[i+1]])
    splitted_prob.append(prob[idx[-1]:])
    
    gen_data = np.zeros((size, 1))
    for k in tqdm(range(len(splitted_prob))):
        cum_prob = np.cumsum(splitted_prob[k])
        
        temp_data = np.zeros((size, len(cum_prob)))
        num = np.random.rand(size)
        for l in range(len(num)):
            for m in range(len(cum_prob)):
                if num[l] < cum_prob[m]:
                    temp_data[l][m] = 1
                    break
                        
        gen_data = np.append(gen_data, temp_data, axis = 1)
    gen_data = np.delete(gen_data, 0, 1)
    
    #===================Generating the continuous part======================# 
    cont_data = np.random.normal(0, 1, size = (size, data_c.shape[1]))
    gen_data = np.append(gen_data, cont_data, axis = 1)
    
    return gen_data
    
        
    