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
# data.to_csv("cleaned_data.csv", index=False)
