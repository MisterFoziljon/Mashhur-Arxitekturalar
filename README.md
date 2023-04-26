## Mashhur-Arxitekturalar

Hozirgi kunda Convolutional Neural Network ning bir nechta modellari ommaga taqdim etilgan. 
![timeline](https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/timeLine.png)

Ushbu modellar juda chuqur(deep) qatlamga ega va yuqori texnik xususiyatlarga ega kompyuterlar yordamida o'qitiladi (GPU,TPU).
Ushbu loyiha tarkibidagi arxitekturalar ImageNet tomonidan o'tkazilgan musobaqalarda g'alaba qozongan mashhur arxitekturalar hisoblanib, ular yordamida 1000 ta sinfga ajratuvchi classification modellarini yaratish mumkin.

<div align="center">
 
| Yil | Arxitektura nomi | Tavsif |
|:----------:|:------------------:|:---------:|
| 2011-2012 | AlexNet | Aleks Krizhevskiy, Ilya Sutskever va Jeffri Xinton tomonidan ishlab chiqilgan konvolyutsion neyron tarmog'i (CNN) |
| 2013 | ZFNet | Nyu-York universitetida Metyu Zayler va Rob Fergus tomonidan ishlab chiqilgan CNN |
| 2014 | GoogleNet (Inception-v1) | Kristian Szegedi va uning Googledagi hamkasblari tomonidan ishlab chiqilgan CNN |
| 2015 | ResNet | Kaiming He va uning Microsoft Research Asiadagi hamkasblari tomonidan ishlab chiqilgan juda chuqur CNN |
| 2016 | ResNet | yanada chuqurroq versiya bilan (ResNet-152) |
| 2017 | Inception-v4 va Inception-ResNet-v2 | Kristian Szegedi va uning Googledagi hamkasblari tomonidan ishlab chiqilgan |
| 2018 | SENet (Squeeze and Excitation Network) | Jie Xu va uning Microsoft Research Asiadagi hamkasblari tomonidan ishlab chiqilgan |
| 2019 | EfficientNet | Google kompaniyasida Mingxing Tan va Quoc V. Le tomonidan ishlab chiqilgan CNN |
| 2020 | Big Transfer (BiT) | Google tomonidan ishlab chiqilgan modellar oilasi boʻlib, oldindan oʻrgatilgan yirik modellardan transfer oʻrganishdan foydalanadi |
| 2021 | ViT (Vision Transformer) | Aleksey Dosovitskiy va uning Googledagi hamkasblari tomonidan ishlab chiqilgan transformatorga asoslangan arxitektura |
</div>

[tensorflow.datasets](https://www.tensorflow.org/datasets/catalog/overview) tarkibidagi [oxford_flowers102](https://www.tensorflow.org/datasets/catalog/oxford_flowers102?hl=ru) dataseti 102 sinfdan iborat. Mana shu dataset yordamida mashhur arxitekturalar sinovdan o'tkazildi. Quyida arxitekturalar haqida qisqacha ma'lumot, modelning qurilishi hamda train va test qilish natijalari haqida ma'lumot olishingiz mumkin.

### 1. VGG-19 arxitekturasi
VGG-19 konvolyutsion neyron tarmog'i (CNN) bo'lib, 19 ta asosiy qatlam (16 ta konvolyutsion, 3 ta to'liq ulangan), shuningdek 5 ta MaxPool qatlami va 1 ta SoftMax qatlamiga ega. U 2014 yilda Oksford universitetida ishlab chiqilgan va o'qitilgan.

VGG-19 tarmog'ini o'qitish uchun ImageNet ma'lumotlar bazasidan 1 milliondan ortiq tasvir ishlatilgan. Tabiiyki, siz ImageNetdan o'qitilgan vaznli modelni import qilishingiz mumkin. Bu oldindan tayyorlangan tarmoq 1000 tagacha obyektni tasniflashi mumkin. Tarmoq 224×224 piksel o'lchamdagi rangli tasvirlarga o'rgatilgan. O'lchami va VGG-19 ishlashi haqida qisqacha ma'lumot:

- Hajmi: 549 MB
- Top-1: 71,3%
- Top-5: 90,0%
- Parametrlar soni: 143 667 240 ta
- Qatlamlarning umumiy soni: 25
<img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/VGG-19.png" width="500" align="center" />

### 2. GoogleNet arxitekturasi
![googlenet](https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/GoogleNet.png)

### 3. ResNet arxitekturasi
![resnet](https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/ResNet.png)

### 4. DenseNet arxitekturasi
![densenet](https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/DenseNet.jpg)
