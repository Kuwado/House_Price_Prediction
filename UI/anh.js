// Hàm gọi khi người dùng nhấn nút "Tìm Kiếm"
function Search() {
    // Lấy giá trị từ các trường nhập
    const location = document.querySelector('.location').value;
    const demand = document.querySelector('.demand').value;
    const propertyType = document.querySelector('.property-type').value;
    const province = document.querySelector('.province').value;
    const district = document.querySelector('.district').value;
    const direct = document.querySelector('.direct').value;
    const square = document.querySelector('.square').value;
    const price = document.querySelector('.price').value;

    // Xây dựng URL tìm kiếm với các tham số
    const searchUrl = `/search-results?location=${encodeURIComponent(location)}&demand=${demand}&propertyType=${propertyType}&province=${province}&district=${district}&direct=${direct}&square=${square}&price=${price}`;

    // Điều hướng tới trang kết quả tìm kiếm
    window.location.href = searchUrl;
}

// Hàm gọi khi tỉnh/thành phố thay đổi để tải danh sách quận/huyện
function LoadQuanHuyen() {
    const provinceId = document.querySelector('.province').value;

    // Kiểm tra nếu không chọn tỉnh/thành
    if (provinceId == 0) return;

    // Gọi API hoặc lấy dữ liệu quận/huyện dựa trên provinceId
    // Ví dụ giả lập với mảng quận/huyện cho từng tỉnh
    const districts = {
        1: ["Ba Đình", "Hoàn Kiếm", "Đống Đa"],
        2: ["Quận 1", "Quận 2", "Quận 3"],
        // Thêm các tỉnh khác với danh sách quận/huyện tương ứng
    };

    const districtDropdown = document.querySelector('.district');
    districtDropdown.innerHTML = '<option value="0">---------- Tất cả ----------</option>'; // Xóa các tùy chọn cũ

    if (districts[provinceId]) {
        districts[provinceId].forEach(district => {
            const option = document.createElement('option');
            option.value = district;
            option.textContent = district;
            districtDropdown.appendChild(option);
        });
    }
}

// Hàm gọi khi nhấp vào trường quận/huyện để làm mới danh sách
function huyen_click() {
    document.querySelector('.location').value = '';
    document.querySelector('.hddStreet').value = 0;
    document.querySelector('.hddWard').value = 0;
}
