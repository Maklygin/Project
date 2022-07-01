# Project

1. [Вступление](#Вступление)
2. [Метод](#Методы)
3. [Результат](#Результат)


### Вступление
Данный репозиторий содержит основные части кода для решения задачи кросс-языкового извлечения информации из биомедицинских текстов (на русском и китайском языках).
Извлекаемая информация - отношения между сущностями, то есть информация о том, каким образом химическое соединение влияет на белок. Структурирование подобной информации
помогает учёным в разработке лекарственных средств и в фундаментальных исследованиях. 
Корпус взят с ресурса [BioCreatice](https://biocreative.bioinformatics.udel.edu/news/corpora/chemprot-corpus-biocreative-vi/). 

Как уже было отмечено, сущностей всего два типа: химические соединия и белки.(В примере выделены зеленым и красным цветом соответственно)

Пример:


<img src="https://github.com/Maklygin/Project/blob/main/example1.png" alt="ex" width="90%"/>

То есть в данном случае лекарственное средство Mibefradil блокирует белок с кодовым названием CACNA1G. 
Взаимодействия делятся на классы (ингибитор это CPR:4). Всего в работе используются 6 классов (выделенные черным цветом + метка отсутствия взаимодействия)

<img src="https://github.com/Maklygin/Project/blob/main/table.png" alt="Table" width="70%"/>

### Метод

1) Создание корпусов на целевых языках: перевод и разметка [[1]](https://arxiv.org/pdf/2109.06798.pdf).
2) Обучение мультиязычной модели для пар языков: En-Ru, En-Zh, All languages.

Архитектура самой модели [[2]](https://academic.oup.com/bioinformatics/article/36/15/4323/5836503?login=false):

<img src="https://github.com/Maklygin/Project/blob/main/model_.png" alt="arch" width="60%"/>


