# Intelligent Placer

## Формулировка задачи 
- На вход программа получается фотографию одного или нескольких различных предметов и многоугольник нарисованный черным маркером на формате А4.
- Требуется определить возможно ли расположить данные предметы внутри заданного многоугольника.
- Результатом работы программы является ответ True, если удается поместить предметы в многоугольник, или False, если не удается поместить предметы в многоугольник.

## Требования

### Общие
- Предметы могут быть только из заранее заданного тренировочного датасета и могут быть расположены на закрепленной белой поверхности
- Все изображения в .jpg формате
- Угол между направлением камеры и перпендикуляром к фотографируемой поверхности должен быть не более 10°
- Устройство для фотографирования тренировочного датасета и для использования должно совпадать
- Изображения цветные, формат .jpg
- Предметы и многоугольник должны целиком быть на фотографии

### К предметам
- Предметы могут встречаться только один раз
- Предметы не могут перекрывать друг друга
- Предметы не могут пересекаться с многоугольником

### К многоугольнику
- Многоугольник выпуклый
- Многоугольник замкнутый
- Многоугольник нарисован черным маркером на белой бумаге

## Примитивы
 [Примитивы](https://github.com/justfriendlyneighbor/intelligent_placer/blob/develop/primitives)