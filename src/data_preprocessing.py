import pandas as pd

def load_data(path):

    df = pd.read_csv(path)

    print(df.shape)

    return df