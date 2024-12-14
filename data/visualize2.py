import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đọc file
data = pd.read_csv('raw_data.csv')
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

#Xóa bản ghi trùng
data=data.drop_duplicates()
print("Số dòng: ", data.shape[0])

# Tỉ lệ giá trị null
print("Tỷ lệ null: ")
print(data.isnull().sum() / len(data) * 100)

# Xóa các bản ghi bị null > 4 cột
data.dropna(thresh=data.shape[1] - 4, inplace=True)
print("Số dòng: ", data.shape[0])

# Số dòng có price null
print("Số dòng price null: ", data[data["Giá/m2"].isnull()].shape[0])
data.dropna(subset=['Giá/m2'], inplace=True)
print("Số dòng: ", data.shape[0])

# Số dòng quận null
print("Số cột quận null: ", data[data["Quận"].isnull()].shape[0])
print(data[data["Quận"].isnull()])
# Nhận thấy số bản ghi null ít -> tự điền quận huyện
data.loc[48418, 'Quận'] = 'Huyện Thanh Trì'
data.loc[48418, 'Phường'] = 'Xã Ngọc Hồi'

# Số dòng phường null
print("Số cột phường null: ", data[data["Phường"].isnull()].shape[0])
print(data[data["Phường"].isnull()])
# Nhận thấy với các bản ghi có địa chỉ => tra được phường, với bản ghi ko có địa chỉ => xóa
data = data[~(data["Phường"].isnull() & data["Địa chỉ"].isnull())]
print("Số dòng: ", data.shape[0])
print(data[data["Phường"].isnull()])
# Nhận thấy có 8 bản ghi thiếu
data.drop(index=324, inplace=True) # Đường An Dương Vương thuộc 2 phường => không xác định được => xóa
data.loc[29300, 'Phường'] = 'Phường Quan Hoa'
data.drop(index=35553, inplace=True)
data.drop(index=54426, inplace=True)
data.loc[62059, 'Phường'] = 'Phường Mai Dịch'
data.drop(index=69475, inplace=True)
data.drop(index=70520, inplace=True)
data.drop(index=81032, inplace=True)

# Số dòng loại nhà ở null
print("Số cột loại nhà ở null: ", data[data["Loại hình nhà ở"].isnull()].shape[0])
print(data[data["Loại hình nhà ở"].isnull()])
# Xóa loại hình nhảf ở null
data = data.dropna(subset=["Loại hình nhà ở"])

# Số dòng giấy tờ null
print("Số cột giấy tờ null: ", data[data['Giấy tờ pháp lý'].isnull()].shape[0])
print(data[data['Giấy tờ pháp lý'].isnull()])
# Hơn 27k => fill là không rõ
data['Giấy tờ pháp lý'].fillna("Không rõ", inplace=True)

# Số dòng giấy tờ null
print("Số cột tầng null: ", data[data['Số tầng'].isnull()].shape[0])
print(data[data['Số tầng'].isnull()])
# Hơn 43k =>
data.loc[data['Số tầng']=='Nhiều hơn 10','Số tầng']='11'
data = data.dropna(subset=["Số tầng"])


# Vẽ đồ thị

# Xử lý cột 'Giá/m2' để loại bỏ đơn vị và chuyển thành số
data['Giá/m2'] = (
    data['Giá/m2']
    .str.replace(r'[^\d.]', '', regex=True)  # Loại bỏ tất cả ký tự không phải số và dấu chấm
    .str.replace(r'(\d)\.(?=\d{3})', r'\1', regex=True)  # Loại bỏ dấu chấm phân cách ngàn
    .astype(float)  # Chuyển thành kiểu float
)

data['Số tầng'] = (
    data['Số tầng']
    .str.replace(r'[^\d]', '', regex=True)  # Loại bỏ tất cả ký tự không phải số
    .astype(int)  # Chuyển thành kiểu float
)
unique_values = data['Số tầng'].unique()
print(unique_values)
unique_values = data['Giá/m2'].unique()
print(unique_values)
# Sắp xếp dữ liệu theo 'Số tầng' tăng dần
data_sorted = data.sort_values(by='Số tầng', ascending=True)

# Cập nhật cột 'Số tầng' để chúng lần lượt từ 1, 2, 3,...
# data_sorted['Số tầng'] = range(1, len(data_sorted) + 1)

# Kiểm tra lại dữ liệu đã được sắp xếp và cập nhật
print(data_sorted['Số tầng'])

# Vẽ đồ thị với dữ liệu đã sắp xếp
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_sorted, x='Số tầng', y='Giá/m2', alpha=0.7)

# Thiết lập nhãn và tiêu đề
plt.title('Mối quan hệ giữa Số tầng và Giá', fontsize=14)
plt.xlabel('Số tầng', fontsize=12)
plt.ylabel('Giá/m2', fontsize=12)
plt.grid(alpha=0.3)

# Hiển thị đồ thị
plt.show()
# plt.bar(data_sorted['Số tầng'], data_sorted['Giá/m2'])  # 'Giá trị' là một cột bất kỳ để minh họa
# plt.xlabel('Số tầng')
# plt.ylabel('Giá/m2')
# plt.title('Biểu đồ số tầng')
# plt.xticks(data_sorted['Số tầng'])  # Đảm bảo trục X hiển thị lần lượt 1, 2, 3, ...
# plt.show()

