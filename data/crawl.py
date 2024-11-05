import requests
from bs4 import BeautifulSoup
import csv
import os

baseUrl = 'https://alonhadat.com.vn/'
startUrl = 'https://alonhadat.com.vn/can-ban-nha-ha-noi-t1.htm'

def getInformation(houseUrl):
    response = requests.get(houseUrl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Kiểm tra xem file đã tồn tại chưa
        file_exists = os.path.isfile('rawdata.csv')

        # Mở file để ghi
        with open('rawdata.csv', mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Nếu file chưa tồn tại, viết header
            if not file_exists:
                writer.writerow(['Loại BDS', 'Pháp lý', 'Sân thượng', 
                                 'Chiều ngang', 'Chiều dài', 'Số lầu', 
                                 'Số phòng ngủ', 'Chổ để xe hơi', 
                                 'Hướng', 'Phòng ăn', 'Đường trước nhà', 
                                 'Nhà bếp', 'Diện tích', 'Địa chỉ tài sản'])

            # Lấy thông tin cần thiết từ bảng chi tiết
            infor_div = soup.find('div', class_='infor')
            if infor_div:
                rows = infor_div.find_all('tr')
                
                # Khởi tạo dictionary để lưu thông tin
                house_info = {
                    'Loại BDS': '',
                    'Pháp lý': '',
                    'Sân thượng': '',
                    'Chiều ngang': '',
                    'Chiều dài': '',
                    'Số lầu': '',
                    'Số phòng ngủ': '',
                    'Chổ để xe hơi': '',
                    'Hướng': '',
                    'Phòng ăn': '',
                    'Đường trước nhà': '',
                    'Nhà bếp': '',
                    'Diện tích': '',
                    'Địa chỉ tài sản': ''
                }
                
                # Lặp qua các dòng trong bảng và lấy thông tin
                for row in rows:
                    cells = row.find_all('td')
                    for i in range(0, len(cells) - 1, 2):
                        label = cells[i].text.strip()
                        value_cell = cells[i+1]
                        # Kiểm tra nếu ô chứa ảnh thì gán giá trị là "có"
                        if value_cell.find('img', {'src': '/publish/img/check.gif'}):
                            value = 'có'
                        else:
                            value = value_cell.text.strip()
                        # Lưu các thông tin vào dictionary nếu label tồn tại
                        if label in house_info:
                            house_info[label] = value

                # Lấy thông tin diện tích
                square_div = soup.find('span', class_='square')
                if square_div:
                    house_info['Diện tích'] = square_div.find('span', class_='value').text.strip()

                # Lấy thông tin địa chỉ tài sản
                address_div = soup.find('div', class_='address')
                if address_div:
                    house_info['Địa chỉ tài sản'] = address_div.find('span', class_='value').text.strip()

                # Ghi thông tin vào file CSV
                writer.writerow([
                    house_info['Loại BDS'],
                    house_info['Pháp lý'],
                    house_info['Sân thượng'],
                    house_info['Chiều ngang'],
                    house_info['Chiều dài'],
                    house_info['Số lầu'],
                    house_info['Số phòng ngủ'],
                    house_info['Chổ để xe hơi'],
                    house_info['Hướng'],
                    house_info['Phòng ăn'],
                    house_info['Đường trước nhà'],
                    house_info['Nhà bếp'],
                    house_info['Diện tích'],
                    house_info['Địa chỉ tài sản']
                ])
                print(f'Đã lưu thông tin cho: {houseUrl}')
            else:
                print(f'Không tìm thấy thông tin cần thiết cho: {houseUrl}')
    else:
        print(f'Không thể truy cập: {houseUrl}, mã trạng thái: {response.status_code}')

def getHouseLinks(listUrl):
    response = requests.get(listUrl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Lấy tất cả các bài viết từ danh sách trang
        for content_item in soup.find_all('div', class_='content-item'):
            ct_title = content_item.find('div', class_='ct_title')
            if ct_title:
                a = ct_title.find('a')
                if a and a.get('href'):
                    houseUrl = baseUrl + a.get('href')
                    getInformation(houseUrl)
    else:
        print(f'Không thể truy cập URL danh sách: {listUrl}, mã trạng thái: {response.status_code}')

# Bắt đầu quá trình lấy dữ liệu từ nhiều trang
def scrapeMultiplePages(startUrl, num_pages):
    for page_num in range(1, num_pages + 1):
        if page_num == 1:
            listUrl = startUrl
        else:
            listUrl = f'{baseUrl}can-ban-nha-ha-noi-t1/trang-{page_num}.htm'
        print(f'Đang crawl trang {page_num}...')
        getHouseLinks(listUrl)

# Gọi hàm để crawl 10 trang
scrapeMultiplePages(startUrl, 10)
