# Intelligent Placer

## Формулировка задачи 
- На вход программа получает фотографию содержащюю многоугольник нарисованный черным маркером на листе формата А4 и один или нескольких различных предметов, лежащие вне листа на заданной поверхности.
- Требуется определить возможно ли расположить данные предметы внутри заданного многоугольника.
- Результатом работы программы является ответ True, если удается поместить предметы в многоугольник, или False, если не удается поместить предметы в многоугольник.

## Требования

### Общие
- Предметы могут быть только из заранее заданного тренировочного датасета и могут быть расположены на закрепленной оранжевой поверхности (той же что была использована при фотографировании тренировочного датасета)
- Угол между направлением камеры и перпендикуляром к фотографируемой поверхности должен быть не более 10°
- Устройство для фотографирования тренировочного датасета и для использования должно совпадать
- Все изображения цветные, формат .jpg
- Предметы и многоугольник должны целиком быть на одной фотографии

### К предметам
- Предметы могут встречаться только один раз
- Предметы не могут перекрывать друг друга или лежать друг на друге или друг в друге
- Предметы не могут пересекаться с многоугольником

### К многоугольнику
- Многоугольник выпуклый
- Многоугольник замкнутый
- Многоугольник нарисован черным маркером на белой бумаге

## Примитивы
 [Примитивы](https://github.com/justfriendlyneighbor/intelligent_placer/tree/develop/Primitives)
 
## План
### 1. Нахождение на фотографии предметов в тестовом датасете
- Для нахождения предметов на листе применить бинаризацию Оцу и морфологические операции (opening, closing и другие) и получить шаблоны.
### 2. Нахождение на фотографии многоугольника и объъектов из тестового датасета
- Для нахождения многоугольника на листе применить фильтр Canny или/и фильтр Гауса. Также найти площадь многоугольника и его бинарную маску.
- Для нахождения предметов провести бинаризацию и сегментацию изображения, сопоставить с шаблонами дескриптором SIFT. Также найти площади предметов.
### 3. Размещение предметов в многоугольнике
- Проверить какие получится требования (например замкнутость и выпуклость многоугольника или непересечение предметов)
- Попробовать размещать предметы по убывания площадей и особенности их формы, или использовать алгоритм для укладки.
