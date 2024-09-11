import pandas as pd

df = pd.read_csv("/Users/shubhan/Downloads/realtor-data.zip.csv")

california_df = df[df['state'] == 'California']

file_path = '/Users/shubhan/Desktop/California_Real_Estate.xlsx'

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
    california_df.to_excel(writer, sheet_name='NewData', index=False)