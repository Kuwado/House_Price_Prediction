import requests
from bs4 import BeautifulSoup
import csv
import os

baseUrl = 'https://alonhadat.com.vn/'
url = 'https://alonhadat.com.vn/can-ban-nha-ha-noi-t1.htm'

u = 'https://alonhadat.com.vn/nha-dat/can-ban/nha-mat-tien/1/ha-noi/trang--3.html' 

print(u)

def getInfomation(houseUrl):
        response = requests.get(houseUrl)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            file_exists = os.path.isfile('rawdata.csv')

            
            with open('rawdata.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Dientich', 'Gia'])
                
                squareSpan = soup.find('span', class_='square')
                square = squareSpan.find('span', class_='value').text
                
                priceSpan = soup.find('span', class_='price')
                price = priceSpan.find('span', class_='value').text
                print(price)
                writer.writerow([square, price])
    
    
def getHouseLinks(listUrl):
    response = requests.get(listUrl)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        houseLinks = []
        for div in soup.find_all('div', class_='ct_title'):
            a = div.find('a')
            houseUrl = a.get('href')
            getInfomation(baseUrl + houseUrl)
            
getHouseLinks(url)


