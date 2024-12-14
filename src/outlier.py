import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Đọc file
data = pd.read_csv("../data/filled_data.csv")

# Số dòng số cột
num_rows, num_cols = data.shape
print("Số dòng: ", num_rows)
print("Số cột: ", num_cols)

# In tên các cột
print("Tên các cột: ", list(data.columns))

# Đếm số dong trùng lặp
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))


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
    bins = np.arange(0, x.max() + 2)  # Tạo các bin từ 0 đến max(Số tầng) + 1
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


# ----------------------------------- Outlier ----------------------------
print("Số dòng: ", data.shape[0])

# --- Số tầng ---
print(data["Số tầng"].unique())
print("Số lượng số tầng: ", data["Số tầng"].unique().shape[0])  # 25
# drawHist(data["Số tầng"], "Số tầng")

print("Số nhà có nhiều hơn 11 tầng: ", data[(data["Số tầng"] > 11)].shape[0])  # 29
data = data[(data["Số tầng"] <= 11)]  # Chỉ lấy <= 11 tầng
print("Số dòng: ", data.shape[0])

# --- Số phòng ngủ ---
print(data["Số phòng ngủ"].unique())
print("Số lượng số phòng ngủ: ", data["Số phòng ngủ"].unique().shape[0])  # 10
# drawHist(data["Số phòng ngủ"], "Số phòng ngủ")

print("Số dòng: ", data.shape[0])  # oke

# --- Quận ---
col = "Phường"
print(data["Quận"].unique())
print("Số lượng Quận: ", data["Quận"].unique().shape[0])  # 26
# drawBarth(data["Quận"], "Quận")
counts = data["Quận"].value_counts()
data = data[data["Quận"].isin(counts[counts >= 100].index)]

# --- Phường ---
# drawBarth(
#     data[(data["Quận"] == "Quận Đống Đa")]["Phường"], "Phường của Quận Đống Đa", 5
# )
# drawBarth(
#     data[(data["Quận"] == "Quận Thanh Xuân")]["Phường"], "Phường của Quận Thanh Xuân", 5
# )
# drawBarth(
#     data[(data["Quận"] == "Quận Hoàng Mai")]["Phường"], "Phường của Quận Hoàng Mai", 5
# )
# drawBarth(
#     data[(data["Quận"] == "Quận Hà Đông")]["Phường"], "Phường của Quận Hà Đông", 5
# )

# --- Loại hình nhà ở---
print(data["Loại hình nhà ở"].unique())
print("Số lượng Loại hình nhà ở: ", data["Loại hình nhà ở"].unique().shape[0])  # 4
# drawBarth(data["Loại hình nhà ở"], "Loại hình nhà ở")

# --- Giấy tờ pháp lý---
print(data["Giấy tờ pháp lý"].unique())
print("Số lượng Giấy tờ pháp lý: ", data["Giấy tờ pháp lý"].unique().shape[0])  # 4
# drawBarth(data["Giấy tờ pháp lý"], "Giấy tờ pháp lý")

# --- Diện tích ---
print("Số dòng: ", data.shape[0])
print("Diện tích bé nhất: ", data["Diện tích"].min())
print("Diện tích lớn nhất: ", data["Diện tích"].max())
print("Diện tích trung bình: ", data["Diện tích"].mean())
# Có 372 nhà có diện tích > 150 => xét <= 150
print("Số nhà có diện tích > 150: ", data[(data["Diện tích"] > 150)].shape[0])  # 371
print("Số nhà có diện tích > 100: ", data[(data["Diện tích"] > 100)].shape[0])  # 915
print("Số nhà có diện tích < 20: ", data[(data["Diện tích"] < 20)].shape[0])  # 389
# drawDT()

data = data[(data["Diện tích"] <= 100) & (data["Diện tích"] >= 20)]
print("Số dòng: ", data.shape[0])

# --- Giá ---
print("Số dòng: ", data.shape[0])
print("Giá bé nhất: ", data["Giá/m2"].min())
print("Giá lớn nhất: ", data["Giá/m2"].max())
print("Giá trung bình: ", data["Giá/m2"].mean())
# Có 282 nhà > 300 => xét <= 300 oke
print("Số nhà có giá > 300: ", data[(data["Giá/m2"] > 300)].shape[0])  # 500
print("Số nhà có giá > 200: ", data[(data["Giá/m2"] > 200)].shape[0])  # 1592
print("Số nhà có giá < 25: ", data[(data["Giá/m2"] < 25)].shape[0])  # 433
# drawPrice(300, 600)
data = data[(data["Giá/m2"] <= 200) & (data["Giá/m2"] >= 25)]

# Loại bỏ outlier giá
data["Lower Bound"] = data.groupby(["Quận"])["Giá/m2"].transform(
    lambda x: x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25))
)
data["Upper Bound"] = data.groupby(["Quận"])["Giá/m2"].transform(
    lambda x: x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
)
data = data[
    (data["Giá/m2"] >= data["Lower Bound"]) & (data["Giá/m2"] <= data["Upper Bound"])
]
print(data[data["Quận"] == "Quận Hà Đông"].head(5))
data = data.drop(["Lower Bound", "Upper Bound"], axis=1)

# Check giá sau khi loại outlier
print("Số dòng: ", data.shape[0])
print("Giá bé nhất: ", data["Giá/m2"].min())
print("Giá lớn nhất: ", data["Giá/m2"].max())
print("Giá trung bình: ", data["Giá/m2"].mean())
# Có 282 nhà > 300 => xét <= 300 oke
print("Số nhà có giá > 300: ", data[(data["Giá/m2"] > 300)].shape[0])  # 59
print("Số nhà có giá > 200: ", data[(data["Giá/m2"] > 200)].shape[0])  # 4939
print("Số nhà có giá < 50: ", data[(data["Giá/m2"] < 50)].shape[0])  # 1455
# drawPrice(25, 200)

# Lưu data
# data.to_csv("../data/cleaned_data.csv", index=False)
