import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Đọc file
data = pd.read_csv("../../data/cleaned_data.csv")
# Xóa 2 cột đầu không cần thiết -> lọc duplicate

# print(data.head(5))
# print("Tỷ lệ null: ")
# print(data.isnull().sum() / len(data) * 100)

# print(data[data["Ngày"] == "2020-08-05"].shape[0])
# print(data[data["Ngày"] == "2020-08-04"].shape[0])
# print(data[data["Ngày"] == "2020-08-03"].shape[0])
# print(data[data["Ngày"] == "2020-08-02"].shape[0])
# print(data[data["Ngày"] == "2020-08-01"].shape[0])

print(data["Địa chỉ"].unique().shape[0])
print(data[data["Diện tích"] > 100].shape[0])
