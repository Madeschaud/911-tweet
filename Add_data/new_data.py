import pandas as pd

def download_data():

    filepath = 'Data/additional_train_dataset.csv'
    data = pd.read_csv(filepath, sep='\t').drop(columns=['lang', 'lang_conf', 'source'])

    # data = data[['event']].replace('disaster_events', 'NaN')
    print(data.head())
