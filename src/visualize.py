import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Đọc file
data = pd.read_csv("../data/cleaned_data.csv")

# Số dòng số cột
num_rows, num_cols = data.shape
print("Số dòng: ", num_rows)  # 20525
print("Số cột: ", num_cols)  # 9

# In tên các cột
print("Tên các cột: ", list(data.columns))

# Đếm số dong trùng lặp
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))

# Tỉ lệ giá trị null
print("Tỷ lệ null: ")
print(data.isnull().sum() / len(data) * 100)


# --- Hàm vẽ ---
def drawBarth(data, name, margin=25):
    x = data.value_counts().keys()
    y = data.value_counts()
    plt.barh(x, y, color="skyblue", edgecolor="black")
    plt.title(f"Số lượng nhà theo {name}", fontsize=16, fontweight="bold")
    plt.ylabel(name)
    plt.xlabel("Số lượng")
    plt.yticks(fontsize=12)
    for i, count in enumerate(y):
        plt.text(count + margin, i, str(count), va="center", fontsize=12)
    plt.show()


def drawHist(data, name):
    x = data
    bins = np.arange(1, x.max() + 2)  # Tạo các bin từ 0 đến max(Số tầng) + 1
    hist, bin_edges = np.histogram(x, bins=bins)
    plt.hist(x, bins=bins, edgecolor="black", color="skyblue")
    plt.title(f"Phân tích {name}")
    plt.ylabel("Số lượng (nhà)")
    plt.xlabel(name)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Tâm của các bin
    for center, count in zip(bin_centers, hist):
        if count > 0:  # Chỉ thêm nếu số lượng > 0
            plt.text(center, count, str(count), ha="center", va="bottom", fontsize=12)
    plt.show()


def drawDT():
    x = data["Diện tích"]
    plt.hist(x, edgecolor="black", color="skyblue", bins=np.arange(0, 150 + 1))
    plt.title("Phân tích diện tích")
    plt.ylabel("Số lượng (nhà)")
    plt.xlabel("Diện tích (m²)")
    plt.show()


def drawPrice(start, end):
    x = data["Giá/m2"]
    plt.hist(x, edgecolor="black", color="skyblue", bins=np.arange(start, end + 1))
    plt.title("Phân tích giá")
    plt.ylabel("Số lượng (nhà)")
    plt.xlabel("Giá (triệu/m²)")
    plt.show()


def srawScatter():
    y = data["Giá/m2"].sort_values(ascending=False)
    x = data["Diện tích"][y.index]
    plt.scatter(x, y)
    # df['Diện tích']
    plt.ylabel("Giá")
    plt.xlabel("Diện tích")
    plt.show()


# ---------------------- Trực quan hóa dữ liệu ----------------------

# 1. Số lượng

# --- Quận ---
print("Số lượng Quận: ", data["Quận"].unique().shape[0])  # 14
print(data["Quận"].unique())
# drawBarth(data["Quận"], "Quận")

# --- Phường ---
print("Số lượng Phường: ", data["Phường"].unique().shape[0])  # 188

# --- Loại hình nhà ở ---
print("Số lượng Loại hình nhà ở: ", data["Loại hình nhà ở"].unique().shape[0])  # 4
print(data["Loại hình nhà ở"].unique())
# drawBarth(data["Loại hình nhà ở"], "Loại hình nhà ở")

# --- Giấy tờ pháp lý ---
print("Số lượng Giấy tờ pháp lý: ", data["Giấy tờ pháp lý"].unique().shape[0])  # 4
print(data["Giấy tờ pháp lý"].unique())
# drawBarth(data["Giấy tờ pháp lý"], "Giấy tờ pháp lý")

# --- Số tầng ---
print(data["Số tầng"].unique())
print("Số lượng số tầng: ", data["Số tầng"].unique().shape[0])  # 9
# drawHist(data["Số tầng"], "Số tầng")

# --- Số phòng ngủ ---
print(data["Số phòng ngủ"].unique())
print("Số lượng Số phòng ngủ: ", data["Số phòng ngủ"].unique().shape[0])  # 10
# drawHist(data["Số phòng ngủ"], "Số phòng ngủ")

# 2. Theo giá
print("Diện tích bé nhất: ", data["Diện tích"].min())
print("Diện tích lớn nhất: ", data["Diện tích"].max())
print("Diện tích trung bình: ", data["Diện tích"].mean())
srawScatter()
