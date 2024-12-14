import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re

# Đọc file
data = pd.read_csv("raw_data.csv")
# Xóa 2 cột đầu không cần thiết -> lọc duplicate
data = data.drop(data.columns[:2], axis=1)
print(data.head(5))

# Số dòng số cột
num_rows, num_cols = data.shape
print("Số dòng: ", num_rows)
print("Số cột: ", num_cols)

# In tên các cột
print("Tên các cột: ", list(data.columns))

# Đếm số dong trùng lặp
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))

# Xóa bản ghi trùng
data = data.drop_duplicates()
print("Số dòng: ", data.shape[0])
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))

# Tỉ lệ giá trị null
print("Tỷ lệ null: ")
print(data.isnull().sum() / len(data) * 100)

# Xóa các bản ghi bị null > 4 cột
data.dropna(thresh=data.shape[1] - 4, inplace=True)
print("Số dòng: ", data.shape[0])

# Số dòng có price null
print("Số dòng price null: ", data[data["Giá/m2"].isnull()].shape[0])
data.dropna(subset=["Giá/m2"], inplace=True)
print("Số dòng: ", data.shape[0])

# Số dòng quận null
print("Số cột quận null: ", data[data["Quận"].isnull()].shape[0])
print(data[data["Quận"].isnull()])
# Nhận thấy số bản ghi null ít -> tự điền quận huyện
data.loc[48418, "Quận"] = "Huyện Thanh Trì"
data.loc[48418, "Phường"] = "Xã Ngọc Hồi"

# Số dòng phường null
print("Số cột phường null: ", data[data["Phường"].isnull()].shape[0])
print(data[data["Phường"].isnull()])
# Nhận thấy với các bản ghi có địa chỉ => tra được phường, với bản ghi ko có địa chỉ => xóa
data = data[~(data["Phường"].isnull() & data["Địa chỉ"].isnull())]
print("Số dòng: ", data.shape[0])
print(data[data["Phường"].isnull()])
# Nhận thấy có 8 bản ghi thiếu
data.drop(
    index=324, inplace=True
)  # Đường An Dương Vương thuộc 2 phường => không xác định được => xóa
data.loc[29300, "Phường"] = "Phường Quan Hoa"
data.drop(index=35553, inplace=True)
data.drop(index=54426, inplace=True)
data.loc[62059, "Phường"] = "Phường Mai Dịch"
data.drop(index=69475, inplace=True)
data.drop(index=70520, inplace=True)
data.drop(index=81032, inplace=True)

# Số dòng loại nhà ở null
print("Số cột loại nhà ở null: ", data[data["Loại hình nhà ở"].isnull()].shape[0])
print(data[data["Loại hình nhà ở"].isnull()])
# Xóa loại hình nhảf ở null
data = data.dropna(subset=["Loại hình nhà ở"])

# Số dòng giấy tờ null
print("Số cột giấy tờ null: ", data[data["Giấy tờ pháp lý"].isnull()].shape[0])
print(data[data["Giấy tờ pháp lý"].isnull()])
# Hơn 27k => fill là không rõ
data["Giấy tờ pháp lý"] = data["Giấy tờ pháp lý"].fillna("Không rõ")

# Số dòng số tầng null
print("Số cột tầng null: ", data[data["Số tầng"].isnull()].shape[0])
print(data[data["Số tầng"].isnull()])
# Hơn 43k => xóa
data = data.dropna(subset=["Số tầng"])
print("Số dòng: ", data.shape[0])

# Số dòng số phòng ngủ null = 0
print("Số cột phòng ngủ null: ", data[data["Số phòng ngủ"].isnull()].shape[0])

# Số dòng dài rộng null
print("Số cột dài null: ", data[data["Dài"].isnull()].shape[0])
print("Số cột rộng null: ", data[data["Rộng"].isnull()].shape[0])
# Không có cơ sở để fill => xóa 2 cột luôn
data = data.drop(columns=["Dài", "Rộng"])
print("Tên các cột: ", list(data.columns))

# Dữ liệu đã hết null
print("Tỷ lệ null: ")
print(data.isnull().sum() / len(data) * 100)
num_rows, num_cols = data.shape
print("Số dòng: ", num_rows)
print("Số cột: ", num_cols)
# => Còn 35677 bản ghi
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))

# Xóa bản ghi trùng
data = data.drop_duplicates()
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))

# Dữ liệu cuối cùng
print("Số dòng: ", data.shape[0])  # 34590
print("Số cột: ", data.shape[1])  # 9
print("Tên các cột: ", list(data.columns))
print("--------------------------------------------------------")

# -------------------------------- Type ---------------------------------------

# Quận
print("Số quận: ", data["Quận"].unique().shape[0])  # 26
# print(data['Quận'].unique())

# Phường
print("Số phường: ", data["Phường"].unique().shape[0])
# print(data['Phường'].unique())

# Loại hình nhà ở
print("Số loại hình nhà ở: ", data["Loại hình nhà ở"].unique().shape[0])
# print(data['Loại hình nhà ở'].unique())

# Giấy tờ pháp lý
print("Số giấy tờ pháp lý: ", data["Giấy tờ pháp lý"].unique().shape[0])
# print(data['Giấy tờ pháp lý'].unique())

# Số tầng
print(data["Số tầng"].unique())
print("Số dòng nhiều hơn 10: ", data[data["Số tầng"] == "Nhiều hơn 10"].shape[0])
# Có 7 dòng => xóa luôn
data = data[data["Số tầng"] != "Nhiều hơn 10"]
data["Số tầng"] = data["Số tầng"].astype(int)
# print(data['Số tầng'].unique())

# Số phòng
print(data["Số phòng ngủ"].unique())
print(
    "Số dòng nhiều hơn 10: ",
    data[data["Số phòng ngủ"] == "nhiều hơn 10 phòng"].shape[0],
)
# có 371 dòng => xóa
data = data[data["Số phòng ngủ"] != "nhiều hơn 10 phòng"]
data["Số phòng ngủ"] = data["Số phòng ngủ"].str.replace("phòng", "")
data["Số phòng ngủ"] = data["Số phòng ngủ"].astype(int)
# print(data['Số phòng ngủ'].unique())

# Diện tích
print("Số Diện tích: ", data["Diện tích"].unique().shape[0])
data["Diện tích"] = data["Diện tích"].str.replace(" m²", "")
data["Diện tích"] = data["Diện tích"].astype(float)
# print(data["Diện tích"].unique().tolist())


# Giá/m2
# Thêm cột đơn vị
data["Đơn vị"] = data["Giá/m2"].apply(lambda x: re.findall(r"[^\d.,]+", x)[0])
print("Đơn vị: ", data["Đơn vị"].unique())

# Đơn vị cần quy đổi
print(data[data["Đơn vị"] == " đ/m²"].shape[0])  # 291
print(data[data["Đơn vị"] == " tỷ/m²"].shape[0])  # 46

# Convert giá
print("Số Giá/m2: ", data["Giá/m2"].unique().shape[0])
data["Giá/m2"] = data["Giá/m2"].str.replace(".", "")
data["Giá/m2"] = data["Giá/m2"].str.replace(",", ".")
data["Giá/m2"] = data["Giá/m2"].str.replace(" triệu/m²", "")
data["Giá/m2"] = data["Giá/m2"].str.replace(" đ/m²", "")
data["Giá/m2"] = data["Giá/m2"].str.replace(" tỷ/m²", "")
data["Giá/m2"] = data["Giá/m2"].astype(float)


# Chuyển đơn vị
def convert_price(row):
    if row["Đơn vị"] == " tỷ/m²":
        return row["Giá/m2"] * 1000
    elif row["Đơn vị"] == " đ/m²":
        return row["Giá/m2"] * 0.000001
    else:
        return row["Giá/m2"]


data["Giá/m2"] = data.apply(convert_price, axis=1)
data = data.drop(columns="Đơn vị")  # Xóa cột đơn vị vì dùng xong rồi kaka
# print(data["Giá/m2"].unique().tolist())
print("--------------------------------------------")

# ----------------------------------- Outlier ----------------------------

# --- Diện tích ---
print("Số dòng: ", data.shape[0])
print("Diện tích bé nhất: ", data["Diện tích"].min())
print("Diện tích lớn nhất: ", data["Diện tích"].max())
print("Diện tích trung bình: ", data["Diện tích"].mean())
# Có 372 nhà có diện tích > 150 => xét <= 150
print("Số nhà có diện tích > 150: ", data[(data["Diện tích"] > 150)].shape[0])  # 372
print("Số nhà có diện tích > 60: ", data[(data["Diện tích"] > 60)].shape[0])  # 4525
print("Số nhà có diện tích < 30: ", data[(data["Diện tích"] < 30)].shape[0])  # 11455

# x = data["Diện tích"]
# plt.hist(x, edgecolor="black", color="skyblue", bins=np.arange(0, 150 + 1))
# plt.title("Phân tích diện tích")
# plt.ylabel("Số lượng (nhà)")
# plt.xlabel("Diện tích (m²)")
# plt.show()
data = data[(data["Diện tích"] <= 60) & (data["Diện tích"] >= 30)]


# --- Giá ---
print("Số dòng: ", data.shape[0])
print("Giá bé nhất: ", data["Giá/m2"].min())
print("Giá lớn nhất: ", data["Giá/m2"].max())
print("Giá trung bình: ", data["Giá/m2"].mean())
# Có 282 nhà > 300 => xét <= 300 oke
print("Số nhà có giá > 300: ", data[(data["Giá/m2"] > 300)].shape[0])  # 282
print("Số nhà có giá > 120: ", data[(data["Giá/m2"] > 120)].shape[0])  # 4939
print("Số nhà có giá < 50: ", data[(data["Giá/m2"] < 50)].shape[0])  # 1455

# x = data["Giá/m2"]
# plt.hist(x, edgecolor="black", color="skyblue", bins=np.arange(0, 300 + 1))
# plt.title("Phân tích giá")
# plt.ylabel("Số lượng (nhà)")
# plt.xlabel("Giá (triệu/m²)")
# plt.show()
data = data[(data["Giá/m2"] <= 120) & (data["Giá/m2"] >= 50)]

# ---------------------------- Final ----------------------------------
print("Số dòng: ", data.shape[0])
# Đếm số dong trùng lặp
print("Số bản ghi trùng lặp: ", sum(data.duplicated()))  # 0

# Lưu data
data.to_csv("cleaned_data.csv", index=False)
