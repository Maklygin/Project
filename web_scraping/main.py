from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import re
import numpy as np
import pandas as pd
import time

#4049

# содержат таблицы: CN110054699, CN105705522
def key_re(elem):
    return elem.span()[0]


def break_en_patent_execution(browser_driver, to_browser_window):
    browser_driver.close()
    browser_driver.switch_to.window(to_browser_window)
    browser_driver.back()


def break_zh_patent_execution(browser_driver):
    browser_driver.back()


# переход на следующую страницу поиска
def next_tab(browser_driver,first_tab_flag):
    if first_tab_flag:
        some_elem = WebDriverWait(browser_driver, 10).until(
            EC.presence_of_element_located((By.ID, "resultListCommandsForm"))
        )

        some_elem = some_elem.find_element(by=By.XPATH, value="//*[@id='resultListCommandsForm']/div/div[2]/a")
        some_elem.click()
        time.sleep(2)
    else:
        some_elem = WebDriverWait(browser_driver, 10).until(
            EC.presence_of_element_located((By.ID, "resultListCommandsForm"))
        )

        some_elem = some_elem.find_element(by=By.XPATH, value="//*[@id='resultListCommandsForm']/div/div[2]/a[2]")
        some_elem.click()
        time.sleep(2)


# Подключение к прокси серверу
## Адреса прокси серверов лежат здесь: https://sslproxies.org/
def set_proxy(browser_driver, address):
    PROXY_STR = address
    options = webdriver.ChromeOptions()
    options.add_argument('--proxy-server=%s' % PROXY_STR)
    chrome = webdriver.Chrome(options=options)
    chrome.get("http://ipinfo.io")
    pass


def to_page(browser_driver, page_number):
    navigator = browser_driver.find_element(by=By.XPATH,value="//*[@id='resultListCommandsForm:invalidPageNumber']")
    navigator.click()

    nav_search = browser_driver.find_element(by=By.CLASS_NAME,value="ps-paginator-modal--input")
    nav_search.send_keys(Keys.BACK_SPACE)
    time.sleep(1)
    nav_search.send_keys(page_number)
    time.sleep(1)
    nav_search.send_keys(Keys.RETURN)
    time.sleep(20)


PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(PATH)

driver.get("https://patentscope.wipo.int/search/ru/result.jsf?_vid=P22-L1Q5KT-93583")

# поиск по заданному запросу
time.sleep(15)
search = driver.find_element(by=By.ID, value="simpleSearchForm:fpSearch:input")
search.send_keys("FP:(c07) AND ZH_AB:* and FP:(protein)")
search.send_keys(Keys.RETURN)

# ждём прогрузки результатов поиска и переходим по ссылке в патенты


data_zh = pd.read_csv('data_zh')
data_en = pd.read_csv('data_en')
flag = True
first = True
patent_index = 3609

while len(data_zh) < 400:

    if first:
        arr = np.arange(9,200)
    else:
        arr = np.arange(200)

    if first:
        time.sleep(20)
        to_page(driver,19)
        first,flag = False,False

    for i in arr:

        if patent_index % 10 == 0 and patent_index > 3610:
            data_zh.to_csv('data_zh', index=False)
            data_en.to_csv('data_en', index=False)
            time.sleep(5)

        elem = WebDriverWait(driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//*[@id='resultListForm:resultTable:" + str(i) + ":patentResult']/div/div/a"))
        )

        title = driver.find_element(by=By.XPATH, value="//*[@id='resultListForm:resultTable:" + str(
            i) + ":patentResult']/div/div/a/span")
        t = title.text[::-1][::-1]
        print(patent_index+1, title.text)
        patent_index+=1
        elem.click()

        # Проверяем наличие перевода патента
        try:
            patent_family = driver.find_element(by=By.LINK_TEXT, value="Семейство патентов")
        except:
            print("Семейство патентов не найдено")
            driver.back()
            continue

        # выбираем раздел описание патента
        try:
            try:
                elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "Описание"))
                )
            except:
                elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "Полный текст"))
                )
            elem.click()
        except:
            break_zh_patent_execution(driver)
            print('Text is not found')
            continue


        # читаем патент
        try:
            article = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//*[@id='detailMainForm:MyTabViewId:descriptionPanel']/div[2]"))
            )
            text = article.text
        except:
            driver.back()
            continue

        Old_window = driver.window_handles[0]

        # Список возможных заголовков
        begin_zh = ["背景技术\n", "发明背景\n","背景领域\n","相关领域的描述\n"]
        middle_zh = ["发明简述\n", "发明内容\n","发明概述\n" ]
        end_zh = ["缩写\n", "附图说明\n", "附图简述\n","附图简要说明\n"," 发明的有益效果\n","具体实施方式\n"]

        begin_zh = [re.search(r'' + heading, text[:3000]) for heading in begin_zh]
        begin = [i for i in begin_zh if i is not None]
        # Вычисляем begin как самый первый заголовок, иначе прерываем обработку патента.
        if not begin:
            break_zh_patent_execution(driver)
            print("ZH begin is empty")
            continue
        begin = min(begin, key=key_re)

        # С остальными частями аналогично.
        # Можно улучшить скорость за счёт изменения пределов поиска
        middle_zh = [re.search(r'' + heading, text[begin.span()[1]:int(len(text) * 0.4)]) for heading in middle_zh]
        middle = [i for i in middle_zh if i is not None]
        if not middle:
            break_zh_patent_execution(driver)
            print("ZH middle is empty")
            continue
        middle = min(middle, key=key_re)

        end_zh = [re.search(r'' + heading, text[middle.span()[1]+begin.span()[1]:int(len(text) * 0.6)]) for heading in end_zh]
        end = [i for i in end_zh if i is not None]
        if not end:
            break_zh_patent_execution(driver)
            print("ZH end is empty")
            continue
        end = min(end, key=key_re)

        abstract_zh = re.search(r'' + begin.group(0) + '(.*)' + middle.group(0),
                                text[begin.span()[0]:middle.span()[1] + begin.span()[1]], re.DOTALL).group(0)

        abstract_zh = [t, abstract_zh, re.search(r'' + middle.group(0) + '(.*)' + end.group(0),
                                                 text[
                                                 middle.span()[0] + begin.span()[1]:end.span()[1] + begin.span()[1] +
                                                                                    middle.span()[1]], re.DOTALL).group(
            0)]

        time.sleep(2)

        # идём в семейство патентов

        patent_family = driver.find_element(by=By.LINK_TEXT, value="Семейство патентов")
        patent_family.click()

        patents = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "ps-patent-result--title"))
        )
        patents = driver.find_elements(by=By.CLASS_NAME, value="ps-patent-result--title")
        patent = None

        try:
            for i in patents:
                if "US" in i.text[0:2]:
                    j = 2
                    while i.text[j].isdigit():
                        j += 1
                    patent = i.find_element(by=By.LINK_TEXT, value=i.text[0:j])
                    patent.click()
                    break
        except:
            break_zh_patent_execution(driver)
            continue

        if patent is not None:
            driver.switch_to.window(driver.window_handles[1])
        else:
            break_zh_patent_execution(driver)
            print("EN patent is missing")
            continue

        # Читаем патент на английском

        try:
            try:
                elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "Описание"))
                )
            except:
                elem = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.LINK_TEXT, "Полный текст"))
                )
        except:
            break_en_patent_execution(driver,Old_window)
            continue

        elem.click()

        try:
            article = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.XPATH, "//*[@id='detailMainForm:MyTabViewId:descriptionPanel']/div[2]"))
            )
            text = article.text

        except:
            break_en_patent_execution(driver, Old_window)
            continue

        begin_en = ["BACKGROUND OF THE INVENTION\n", "BACKGROUND ART\n", "BACKGROUND\n","STATE OF THE ART\n","PRIOR ARTS\n"]
        middle_en = ["DESCRIPTION OF THE INVENTION\n", "SUMMARY\n", "SUMMARY OF THE INVENTION\n","DISCLOSURE OF INVENTION\n"]
        end_en = ["DESCRIPTION OF FIGURES\n", "BRIEF DESCRIPTION OF THE", "ABBREVIATIONS\n","LIST OF FIGURES\n",
                  "FIGURES\n","DETAILED DESCRIPTION\n","BENEFICIAL EFFECT OF THE INVENTION\n"]

        begin_en = [re.search(r'' + heading, text[:3000]) for heading in begin_en]
        begin = [i for i in begin_en if i is not None]
        # Вычисляем begin как самый первый заголовок, иначе прерываем обработку патента.
        if not begin:
            break_en_patent_execution(driver, Old_window)
            print("EN begin is empty")
            continue
        begin = min(begin, key=key_re)

        # С остальными частями аналогично.
        # Можно улучшить скорость за счёт изменения пределов поиска
        middle_en = [re.search(r'' + heading, text[begin.span()[1]:20000]) for heading in middle_en]
        middle = [i for i in middle_en if i is not None]
        if not middle:
            break_en_patent_execution(driver, Old_window)
            print("EN middle is empty")
            continue
        middle = min(middle, key=key_re)

        end_en = [re.search(r'' + heading, text[middle.span()[1]+begin.span()[1]:40000]) for heading in end_en]
        end = [i for i in end_en if i is not None]
        if not end:
            break_en_patent_execution(driver, Old_window)
            print("EN end is empty")
            continue
        end = min(end, key=key_re)

        abstract_en = re.search(r'' + begin.group(0) + '(.*)' + middle.group(0),
                             text[begin.span()[0]:middle.span()[1] + begin.span()[1]], re.DOTALL).group(0)

        abstract_en = [t, abstract_en, re.search(r'' + middle.group(0) + '(.*)' + end.group(0),
                                           text[middle.span()[0] + begin.span()[1]:end.span()[1] + begin.span()[1] +
                                                                                   middle.span()[1]], re.DOTALL).group(0)]

        data_en = data_en.append({'index':abstract_en[0],'abstr_p1':abstract_en[1],'abstr_p2':abstract_en[2]}, ignore_index=True)
        print('EN success')
        data_zh = data_zh.append({'index': abstract_zh[0], 'abstr_p1': abstract_zh[1], 'abstr_p2': abstract_zh[2]},
                                 ignore_index=True)
        print('ZH success')

        driver.close()
        driver.switch_to.window(Old_window)
        driver.back()
        time.sleep(5)

    next_tab(driver, flag)
    flag = False


data_zh.to_csv('data_zh',index=False)
data_en.to_csv('data_en',index=False)


time.sleep(10)
driver.quit()
