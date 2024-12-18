import matplotlib.pyplot as plt
import numpy as np

# Tạo dữ liệu mẫu
np.random.seed(42)
n_samples = 1000

# Giá thực tế (random)
actual_prices = np.random.randint(100000, 1000000, n_samples)

# Giá dự đoán (thêm nhiễu so với giá thực)
predicted_prices = actual_prices + np.random.randint(-50000, 50000, n_samples)

# Chỉ số nhà
houses = np.arange(1, n_samples + 1)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))

# Vẽ giá thực và giá dự đoán
plt.plot(houses, actual_prices, label="Actual Prices", color="blue")
plt.plot(houses, predicted_prices, label="Predicted Prices", color="orange")

# Thêm nhãn và tiêu đề
plt.xlabel("Houses")
plt.ylabel("Prices")
plt.title("Comparison of Actual Prices and Predicted Prices (1000 records)")
plt.legend()

# Hiển thị biểu đồ
plt.tight_layout()
plt.show()
