from omegaconf import OmegaConf
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(config):
    print("Preparing data")
    df = pd.read_csv(config.data.csv_file_path)
    df['label'] = df['sentiment'].apply(lambda x: 0 if x == 'positive' else 1)
    print(df.head())
    print("Done")
    test_size = config.data.test_set_ratio
    train_df, test_df = train_test_split(df, test_size=test_size, stratify= df['sentiment'], random_state= 1234)
    train_df.to_csv(config.data.train_csv_path, index=False)   
    test_df.to_csv(config.data.test_csv_path, index=False)
    
    



if __name__ == '__main__':
    config = OmegaConf.load("./params.yaml")
    prepare_data(config)