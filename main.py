from src.data_preprocessing import load_data

df = load_data(
    "data/raw/IT_SystemPerformanceAndResourceMetricsDataset.csv"
)

print(df.head())