from src.data_preprocessing import load_dataset

df = load_dataset(
    "data/raw/IT_SystemPerformanceAndResourceMetricsDataset.csv"
)

print(df.head())