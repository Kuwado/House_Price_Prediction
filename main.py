import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

pd.set_option('display.width', 2000)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)


data = pd.read_csv("VN_housing_dataset.csv", encoding='utf-8')

selected_columns = data.iloc[:, [2, 5, 7, 8, 9, 12]]  # Chỉ số cột bắt đầu từ 0
print(selected_columns.head(100))  # In 100 dòng đầu tiên
