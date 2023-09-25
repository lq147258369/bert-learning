import requests
import re
import random
import time
import os
import pandas as pd
from bs4 import BeautifulSoup

# 设置请求头
def get_headers():
    user_agent = [
        "Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; rv:38.0) Gecko/20100101 Firefox/38.0",
        "Mozilla/5.0 (Windows NT 10.0; WOW64; Trident/7.0; .NET4.0C; .NET4.0E; .NET CLR 2.0.50727; .NET CLR 3.0.30729; .NET CLR 3.5.30729; InfoPath.3; rv:11.0) like Gecko",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)",
        "Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0; Trident/4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 6.0)",
        "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1",
        "Opera/9.80 (Macintosh; Intel Mac OS X 10.6.8; U; en) Presto/2.8.131 Version/11.11",
        "Opera/9.80 (Windows NT 6.1; U; en) Presto/2.8.131 Version/11.11",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Maxthon 2.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; TencentTraveler 4.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; The World)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SE 2.X MetaSr 1.0; SE 2.X MetaSr 1.0; .NET CLR 2.0.50727; SE 2.X MetaSr 1.0)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; 360SE)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Avant Browser)",
        "Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)",
        "Mozilla/5.0 (iPhone; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
        "Mozilla/5.0 (iPod; U; CPU iPhone OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
        "Mozilla/5.0 (iPad; U; CPU OS 4_3_3 like Mac OS X; en-us) AppleWebKit/533.17.9 (KHTML, like Gecko) Version/5.0.2 Mobile/8J2 Safari/6533.18.5",
        "Mozilla/5.0 (Linux; U; Android 2.3.7; en-us; Nexus One Build/FRF91) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
        "MQQBrowser/26 Mozilla/5.0 (Linux; U; Android 2.3.7; zh-cn; MB200 Build/GRJ22; CyanogenMod-7) AppleWebKit/533.1 (KHTML, like Gecko) Version/4.0 Mobile Safari/533.1",
        "Opera/9.80 (Android 2.3.4; Linux; Opera Mobi/build-1107180945; U; en-GB) Presto/2.8.149 Version/11.10",
        "Mozilla/5.0 (Linux; U; Android 3.0; en-us; Xoom Build/HRI39) AppleWebKit/534.13 (KHTML, like Gecko) Version/4.0 Safari/534.13",
        "Mozilla/5.0 (BlackBerry; U; BlackBerry 9800; en) AppleWebKit/534.1+ (KHTML, like Gecko) Version/6.0.0.337 Mobile Safari/534.1+",
        "Mozilla/5.0 (hp-tablet; Linux; hpwOS/3.0.0; U; en-US) AppleWebKit/534.6 (KHTML, like Gecko) wOSBrowser/233.70 Safari/534.6 TouchPad/1.0",
        "Mozilla/5.0 (SymbianOS/9.4; Series60/5.0 NokiaN97-1/20.0.019; Profile/MIDP-2.1 Configuration/CLDC-1.1) AppleWebKit/525 (KHTML, like Gecko) BrowserNG/7.1.18124",
        "Mozilla/5.0 (compatible; MSIE 9.0; Windows Phone OS 7.5; Trident/5.0; IEMobile/9.0; HTC; Titan)",
        "UCWEB7.0.2.37/28/999",
        "NOKIA5700/ UCWEB7.0.2.37/28/999",
        "Openwave/ UCWEB7.0.2.37/28/999",
        "Mozilla/4.0 (compatible; MSIE 6.0; ) Opera/UCWEB7.0.2.37/28/999",
        "Mozilla/6.0 (iPhone; CPU iPhone OS 8_0 like Mac OS X) AppleWebKit/536.26 (KHTML, like Gecko) Version/8.0 Mobile/10A5376e Safari/8536.25",
        'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.198 Safari/537.36'
    ]

    headers = {
        'Cookie': '_trs_uv=kfp3v12j_6_8t0e; SF_cookie_1=37059734; _trs_ua_s_1=kfxdjigi_6_4w48',
        'Host': 'www.stats.gov.cn',
        'Referer': 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/',
        'User-Agent': random.choice(user_agent),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9'
    }

    return headers
# 获取31省
def get_province():
    url = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/index.html'
    response = requests.get(url, headers=get_headers())
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    #response.encoding = 'GBK'
    page_text = response.text
    soup = BeautifulSoup(page_text, 'lxml')
    all_province = soup.find_all('tr', class_='provincetr')
    province_str = ""  # 为了方便处理，把省份数据变成一个字符串
    for i in range(len(all_province)):
        province_str = province_str + str(all_province[i])
    province = {}
    province_soup = BeautifulSoup(province_str, 'lxml')
    province_href = province_soup.find_all("a")  # 获取所有的a标签
    for i in province_href:
        href_str = str(i)
      # print(href_str)
      # 创建省份数据字典
        province.update({BeautifulSoup(href_str, 'lxml').find("a").text: BeautifulSoup(href_str, 'lxml').find("a")["href"]})
      # print(province)
    response.close()
    # pattern = re.compile("<a href='(.*?)'>(.*?)<")
    # result = list(set(re.findall(pattern, response.text)))
    result=list(province)
    return province

# 写入到csv文件
def write_province():
    province = get_province()
    tem = []
    for i in province:
        tem.append([i, province[i]])
    df_province = pd.DataFrame(tem)
    df_province.to_csv('省.csv', index=0)
    return None


# 获取31省
write_province()
province = pd.read_csv('省.csv').values

# 获取342城市
def get_city(province_code):
    url = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + province_code
    headers=get_headers()
    headers['Referer'] = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/index.html'
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    #response.encoding = 'gbk'
    page_text = response.text
    soup = BeautifulSoup(page_text, 'lxml')
    all_city = soup.find_all('tr', class_='citytr')
    city_str = ""  # 为了方便处理，把省份数据变成一个字符串
    for i in range(len(all_city)):
        city_str = city_str + str(all_city[i])
    city = {}
    city_soup = BeautifulSoup(city_str, 'lxml')
    city_href = city_soup.find_all("a")  # 获取所有的a标签
    for i in city_href:
        href_str = str(i)
      # print(href_str)
      # 创建city数据字典
        city.update({BeautifulSoup(href_str, 'lxml').find("a").text: BeautifulSoup(href_str, 'lxml').find("a")["href"]})
    response.close()
    # pattern = re.compile("<a href='(.*?)'>(.*?)<")
    # result = list(set(re.findall(pattern, response.text)))
    # res = []
    # for j in result:
    #     if '0' not in j[1]:
    #         res.append(j)
    new_city = {}
    for k,v in city.items():
        new_city.setdefault(v, []).append(k)
    return new_city

def write_city():
    tem = []
    for i in province:
        if i[0] == '新疆维吾尔自治区':
            city = get_city(i[1])
            print('正在抓取：' , i[1], '共{}个城市'.format(len(city)))
            time.sleep(random.random())
            for key,v in city.items():
                tem.append([i[0], key, v[0], v[1]])
    pd.DataFrame(tem).to_csv('市.csv', index=0)
    return None

# 获取3068区县
def get_district(city_code):
    url = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + city_code
    headers=get_headers()
    headers['Referer'] = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/{}.html'.format(city_code.split('/')[0])
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    # response.encoding = 'gbk'
    page_text = response.text
    soup = BeautifulSoup(page_text, 'lxml')
    all_district = soup.find_all('tr', class_='countytr')
    district_str = ""  # 为了方便处理，把省份数据变成一个字符串
    for i in range(len(all_district)):
        district_str = district_str + str(all_district[i])
    district = {}
    district_soup = BeautifulSoup(district_str, 'lxml')
    district_href = district_soup.find_all("a")  # 获取所有的a标签
    for i in district_href:
        href_str = str(i)
        # print(href_str)
        # 创建city数据字典
        district.update({BeautifulSoup(href_str, 'lxml').find("a").text: BeautifulSoup(href_str, 'lxml').find("a")["href"]})
    response.close()
    # pattern = re.compile("<a href='(.*?)'>(.*?)<")
    # result = list(set(re.findall(pattern, response.text)))
    # res = []
    # for j in result:
    #     if '0' not in j[1]:
    #         res.append(j)
    new_district = {}
    for k, v in district.items():
        new_district.setdefault(v, []).append(k)
    return new_district

def write_district():
    tem = []
    for i in city:
        district = get_district(i[1])
        print('正在抓取：', i[1], i[3], '共{}个区'.format(len(district)))
        time.sleep(random.random())
        for key,v in district.items():
            tem.append([i[0], i[1], i[2], i[3], key, v[0],  v[1]])
        print(tem[-1], '\n')
    pd.DataFrame(tem).to_csv('区.csv', index=0)
    return None


# 获取43027街道
def get_road(province_code, city_code, district_code):
    url = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + province_code.split('/')[0] + '/' + district_code
    headers=get_headers()
    headers['Referer'] = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + city_code
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    response.encoding = response.apparent_encoding
    # response.encoding = 'gbk'
    page_text = response.text
    soup = BeautifulSoup(page_text, 'lxml')
    all_road = soup.find_all('tr', class_='towntr')
    road_str = ""  # 为了方便处理，把省份数据变成一个字符串
    for i in range(len(all_road)):
        road_str = road_str + str(all_road[i])
    road = {}
    road_soup = BeautifulSoup(road_str, 'lxml')
    road_href = road_soup.find_all("a")  # 获取所有的a标签
    for i in road_href:
        href_str = str(i)
        # print(href_str)
        # 创建city数据字典
        road.update({BeautifulSoup(href_str, 'lxml').find("a").text: BeautifulSoup(href_str, 'lxml').find("a")["href"]})
    response.close()
    # response.encoding = 'gbk'
    # response.close()
    # pattern = re.compile("<a href='(.*?)'>(.*?)<")
    # result = list(set(re.findall(pattern, response.text)))
    # res = []
    # for j in result:
    #     if '0' not in j[1]:
    #         res.append(j)
    new_road = {}
    for k, v in road.items():
        new_road.setdefault(v, []).append(k)
    return new_road

def write_road():
    tem = []
    for i in district:
        if i[-1] in ['莎车县', '叶城县']:
            success = False
            while not success:
                try:
                    road = get_road(i[1], i[4], i[4])
                    print(i[1], i[3], i[5], '爬取成功，共{}个街道'.format(len(road)))
                    time.sleep(random.random() / 2)
                    success = True
                except Exception as e:
                    print(e)
                    print(i[1], i[3], i[5], '爬取失败，重新爬取')
            for key,v in road.items():
                tem.append([i[0], i[1], i[2], i[3], i[4], i[5], key, v[0], v[1]])
            print(tem[-1], '\n')
    pd.DataFrame(tem).to_csv('四级乡镇.csv', index=0)
    return None

# 获取342城市
write_city()
city = pd.read_csv('市.csv').values

# 获取3068区县
write_district()
district = pd.read_csv('区.csv').values

# 获取43027街道
write_road()
df = pd.read_csv('四级乡镇.csv')



# 获取656781五级地址
def get_community(province_code, district_code, road_code):
    url = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + province_code.split('/')[0] + '/' + district_code.split('/')[0] + '/' + road_code
    headers=get_headers()
    headers['Referer'] = 'http://www.stats.gov.cn/sj/tjbz/tjyqhdmhcxhfdm/2022/' + province_code.split('/')[0] + '/' + district_code
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    response.encoding = response.apparent_encoding
    page_text = response.text
    soup = BeautifulSoup(page_text, 'lxml')
    all_community = soup.find_all('tr', class_='villagetr')
    community_str = ""  # 为了方便处理，把省份数据变成一个字符串
    for i in range(len(all_community)):
        community_str = community_str + str(all_community[i])
    community = {}
    community_soup = BeautifulSoup(community_str, 'lxml')
    community_href = community_soup.find_all("td")  # 获取所有的a标签
    def to_matrix(l, n):
        return [l[i:i + n] for i in range(0, len(l), n)]

    community_href = to_matrix(community_href,3)
    for i in community_href:
        href_str = [str(item) for item in i]
        # print(href_str)
        # 创建city数据字典
        a=BeautifulSoup(href_str[0], 'lxml').find("td").text
        community.update({BeautifulSoup(href_str, 'lxml').find("td").text: BeautifulSoup(href_str, 'lxml').find("td")})
    response.close()
    # response.encoding = 'gbk'
    # response.close()
    # pattern = re.compile('<td>(.*?)</td>')
    # result = list(set(re.findall(pattern, response.text)))
    # res = []
    # for j in result:
    #     if not re.findall('^\d*$', j):
    #         res.append(j)
    # res.remove('名称')

    new_community = {}
    a=str(a)
    b=BeautifulSoup(href_str, 'lxml').find("td").text
    c=BeautifulSoup(href_str, 'lxml').find("td")
    for k, v in community.items():
        new_community.setdefault(v, []).append(k)
    return new_community

def write_community(filename):
    tem = []
    for i in road:
        success = False
        while not success:
            try:
                community = get_community(i[1], i[4], i[6])
                print(i[1], i[3], i[5], i[7], '\t------>爬取成功，共{}个村委会'.format(len(community)))
                time.sleep(random.random() / 4)
                success = True
            except Exception as e:
                print(e)
                print(i[1], i[3], i[5], i[7], '\t------>爬取失败，重新爬取')
        for j in community:
            tem.append([i[1],i[3],i[5],i[7], j])
        # print(tem[-1], '\n')
    pd.DataFrame(tem).to_csv(filename, index=0)
    return None

# 合并各省五级地址
def merge():
    file_list = os.listdir('address/')
    data = pd.DataFrame()
    for i in file_list:
        data = data.append(pd.read_csv('address/' + i))
    data.rename(columns={'0':'一级', '1':'二级', '2':'三级', '3':'四级', '4':'五级', }, inplace=True)
    return data

# 分省获取656781五级地址
lis = df['1'].unique()
for i in lis:
    road = df[df['1']==i].values
    write_community(i + '.csv')

# 合并各省五级地址
address = merge()
address.to_csv('address.csv', index=0)
address.head()


