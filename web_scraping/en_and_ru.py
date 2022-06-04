from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.select import Select

import re
import numpy as np
import pandas as pd
import time

def key_re(elem):
    return elem.span()[0]


def break_en_patent_execution(browser_driver, to_browser_window):
    browser_driver.close()
    browser_driver.switch_to.window(to_browser_window)
    browser_driver.back()


def break_ru_patent_execution(browser_driver):
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


def to_page(browser_driver, page_number):
    navigator = browser_driver.find_element(by=By.XPATH,value="//*[@id='resultListCommandsForm:invalidPageNumber']")
    navigator.click()

    nav_search = browser_driver.find_element(by=By.CLASS_NAME,value="ps-paginator-modal--input")
    nav_search.send_keys(Keys.BACK_SPACE)
    time.sleep(1)
    nav_search.send_keys(page_number)
    time.sleep(1)
    nav_search.send_keys(Keys.RETURN)
    time.sleep(10)

def restart(browser_driver,page_number,error):
    PATH = 'C:\Program Files (x86)\chromedriver.exe'
    if error:
        browser_driver = webdriver.Chrome(PATH)
    browser_driver.get("https://patentscope.wipo.int/search/ru/")
    time.sleep(5)
    search = driver.find_element(by=By.ID, value="simpleSearchForm:fpSearch:input")
    search.send_keys("FP:(c07) AND RU_AB:* and FP:(protein)")
    search.send_keys(Keys.RETURN)
    time.sleep(10)
    reshape_search(browser_driver)
    time.sleep(10)
    to_page(browser_driver,page_number)

def reshape_search(browser_driver):
    shape = Select(browser_driver.find_element(by=By.ID,value="resultListCommandsForm:perPage:input"))
    shape.select_by_index(3)



PATH = 'C:\Program Files (x86)\chromedriver.exe'
driver = webdriver.Chrome(PATH)

# ждём прогрузки результатов поиска и переходим по ссылке в патенты

flag = True
first = True

# data_ru = pd.DataFrame(columns=['index','abstr_p1','abstr_p2'])
# data_en = pd.DataFrame(columns=['index','abstr_p1','abstr_p2'])
data_ru = pd.read_csv('patents_ru_and_en/data_ru1')
data_en = pd.read_csv('patents_ru_and_en/data_en1')


#1840
patent_index = 2166
page = 11
ban = [1659,1693,1705,1840,1962]

restart(driver,page,error=False)

while len(data_ru) < 400:

    if first:
        arr = np.arange(patent_index%200,200)
        first,flag = False,False
    else:
        arr = np.arange(200)

    for i in arr:
        if patent_index % 10 == 0 and patent_index > 2170:
            data_ru.to_csv('patents_ru_and_en/data_ru1', index=False)
            data_en.to_csv('patents_ru_and_en/data_en1', index=False)
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

        if patent_index in ban:
            continue
        try:
            elem.click()
        except:
            driver.quit()
            driver = restart(driver,page,error=True)
            continue

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
            break_ru_patent_execution(driver)
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
        # Удаляем цифры перед началом нового абзаца
        text = re.sub(r'\[\d{1,}\]',"",text)

        time.sleep(2)

        # Список возможных заголовков
        begin_ru = ["Уровень техники\n","ПРЕДШЕСТВУЮЩИЙ УРОВЕНЬ ТЕХНИКИ\n","Предпосылки изобретения\n",
                    "Предпосылки создания изобретения\n"]
        middle_ru = ["СУЩНОСТЬ ИЗОБРЕТЕНИЯ\n","СУЩНОСТИ ИЗОБРЕТЕНИЯ\n","Уровень техники изобретения\n",
                     "содержание изобретения\n","ОПИСАНИЕ ИЗОБРЕТЕНИЯ"]
        end_ru = ["ПОДРОБНОЕ ОПИСАНИЕ ИЗОБРЕТЕНИЯ\n","ОПИСАНИЕ ГРАФИЧЕСКИХ МАТЕРИАЛОВ\n","Определения\n",
                  "ОПИСАНИЕ ФИГУР\n","КРАТКОЕ ОПИСАНИЕ ИЛЛЮСТРАЦИЙ\n","описание чертежей\n",
                  "Подробное изобретение\n","Терминология\n"]

        begin_ru = [re.search(r'' + heading, text[:3000],re.IGNORECASE) for heading in begin_ru]
        begin = [i for i in begin_ru if i is not None]
        # Вычисляем begin как самый первый заголовок, иначе прерываем обработку патента.
        if not begin:
            break_ru_patent_execution(driver)
            print("RU begin is empty")
            continue
        begin = min(begin, key=key_re)

        # С остальными частями аналогично.
        # Можно улучшить скорость за счёт изменения пределов поиска
        middle_ru = [re.search(r'' + heading, text[begin.span()[1]:int(len(text) * 0.5)],re.IGNORECASE) for heading in middle_ru]
        middle = [i for i in middle_ru if i is not None]
        if not middle:
            break_ru_patent_execution(driver)
            print("RU middle is empty")
            continue
        middle = min(middle, key=key_re)

        end_ru = [re.search(r'' + heading, text[middle.span()[1] + begin.span()[1]:int(len(text) * 0.7)],re.IGNORECASE) for heading in end_ru]
        end = [i for i in end_ru if i is not None]
        if not end:
            break_ru_patent_execution(driver)
            print("RU end is empty")
            continue
        end = min(end, key=key_re)

        abstract_ru = re.search(r'' + begin.group(0) + '(.*)' + middle.group(0),
                                text[begin.span()[0]:middle.span()[1] + begin.span()[1]],re.DOTALL).group(0)

        abstract_ru = [t, abstract_ru, re.search(r'' + middle.group(0) + '(.*)' + end.group(0),
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
        time.sleep(2)
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
            break_ru_patent_execution(driver)
            continue

        if patent is not None:
            driver.switch_to.window(driver.window_handles[1])
        else:
            break_ru_patent_execution(driver)
            print("EN patent is missing")
            continue

        # Читаем патент на английском
        time.sleep(2)
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

        begin_en = ["BACKGROUND TO THE INVENTION", "BACKGROUND ART", "BACKGROUND\n","STATE OF THE ART","PRIOR ART"]
        middle_en = ["DESCRIPTION OF THE INVENTION\n", "SUMMARY\n", "SUMMARY OF THE INVENTION\n",
                     "DISCLOSURE OF INVENTION\n","DISCLOSURE\n"]
        end_en = ["DESCRIPTION OF FIGURES\n", "BRIEF DESCRIPTION OF THE", "ABBREVIATIONS\n","LIST OF FIGURES\n",
                  "DETAILED DESCRIPTION","OF DRAWINGS\n","DRAWING\n","BENEFICIAL EFFECT OF THE INVENTION\n"]

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
        data_ru = data_ru.append({'index': abstract_ru[0], 'abstr_p1': abstract_ru[1], 'abstr_p2': abstract_ru[2]},
                                 ignore_index=True)
        print('RU success')

        driver.close()
        driver.switch_to.window(Old_window)
        driver.back()
        time.sleep(5)

    page+=1
    next_tab(driver, flag)
    flag = False



time.sleep(10)
driver.quit()