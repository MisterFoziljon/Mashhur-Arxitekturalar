## Mashhur-Arxitekturalar

Hozirgi kunda Convolutional Neural Network ning bir nechta modellari ommaga taqdim etilgan. 

<p align="center">
 <img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/timeLine.png" width="800"/>
</p>

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

- Parametrlar soni: 143,769,342 ta
- Qatlamlarning umumiy soni: 25

<p align="center">
<img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/VGG-19.png" width="500"/>
</p>

```VGG-19.ipynb``` notebookda arxitektura oxford_flowers102 dataseti uchun moslangan va noldan qurilgan. Train va Test qilingan.



### 2. GoogleNet arxitekturasi

GoogLeNet - bu ko'p qatlamli chuqur konvolyutsion neyron tarmoq bo'lib, u Google tadqiqotchilari tomonidan ishlab chiqilgan Inception Network, chuqur konvolyutsion neyron tarmog'ining bir variantidir.,GoogLeNet arxitekturasi 22 qatlamdan iborat (27 qatlam, shu jumladan birlashtiruvchi qatlamlar) va bu qatlamlarning bir qismi jami 9 ta boshlang'ich moduldan iborat.

ImageNet Large-Scale Visual Recognition Challenge 2014 (ILSVRC14) da taqdim etilgan GoogLeNet arxitekturasi tasvirlarni tasniflash va obyektni aniqlash kabi kompyuterni ko'rish vazifalarini hal qildi — uning qanchalik yaxshi ishlashini ushbu [maqola](https://towardsdatascience.com/deep-learning-googlenet-explained-de8861c82765)dan bilib olishingiz mumkin:

Bugungi kunda GoogLeNet kompyuterni ko'rishning boshqa vazifalari uchun ishlatiladi, masalan, yuzni aniqlash va tanib olish va boshqalar.
GoogLeNet tarmog'ini o'qitish uchun ImageNet ma'lumotlar bazasidan 1 milliondan ortiq tasvir ishlatilgan. Tabiiyki, siz ImageNetdan o'qitilgan vaznli modelni import qilishingiz mumkin. Bu oldindan tayyorlangan tarmoq 1000 tagacha obyektni tasniflashi mumkin. Tarmoq 224×224 piksel o'lchamdagi rangli tasvirlarga o'rgatilgan. O'lchami va GoogLeNet ishlashi haqida qisqacha ma'lumot:

- Parametrlar soni: 9 804 474 ta (oxford_flowers102)
- Qatlamlarning umumiy soni: 22

<p align="center">
 <img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/GoogleNet.png" width="800"/>
</p>

```GoogleNet.ipynb``` notebookda arxitektura oxford_flowers102 dataseti uchun moslangan va noldan qurilgan. Train va Test qilingan.



### 3. ResNet arxitekturasi

ResNet34 konvolyutsion neyron tarmoq arxitekturasi bo'lib, u 2015-yilda ResNet modellar oilasining bir qismi sifatida taqdim etilgan bo'lib, u "qoldiq tarmoq" degan ma'noni anglatadi. ResNet34 shunday nomlanadi, chunki u 34 qatlamdan, jumladan konvolyutsion qatlamlar, birlashtiruvchi qatlamlar va to'liq bog'langan qatlamlardan iborat.

ResNet modellarining asosiy yangiligi qatlamlar orasidagi "qoldiq ulanishlar" yoki "o'tkazib yuborilgan ulanishlar" ni joriy etishdir. Ushbu ulanishlar tarmoqqa asl funksiyalarga qaraganda optimallashtirish osonroq bo'lgan qoldiq funktsiyalarni o'rganish imkonini beradi. Bu yo'qolgan gradientlar muammosini hal qilishga yordam beradi va tarmoqni yanada samarali o'qitishga imkon beradi.

ResNet34, xususan, har xil filtr o'lchamlariga ega bo'lgan konvolyutsion qatlamlar to'plamidan iborat bo'lib, bundan tashqari normallashtirish va faollashtirish funktsiyalari mavjud. Har ikki konvolyutsion qatlamdan so'ng, keyingi ikki qatlamning kirishiga qoldiq ulanish qo'shiladi. Tarmoq global o'rtacha birlashtiruvchi qatlam bilan tugaydi, undan keyin tasniflash bashoratlarini chiqaradigan to'liq bog'langan qatlam.

ResNet34 tasvirni tasniflash, ob'ektlarni aniqlash va semantik segmentatsiyani o'z ichiga olgan kompyuterni ko'rishning keng ko'lamli vazifalari uchun ishlatilgan va bir nechta mezonlarda eng zamonaviy ko'rsatkichlarga erishgan.

- Parametrlar soni: 25 738 814 ta (oxford_flowers102)
- Qatlamlarning umumiy soni: 34

<p align="center">
 <img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/ResNet.png" width="800"/>
</p>

```ResNet.ipynb``` notebookda arxitektura oxford_flowers102 dataseti uchun moslangan va noldan qurilgan. Train va Test qilingan.



### 4. DenseNet arxitekturasi



- Parametrlar soni: 7 147 942 ta (oxford_flowers102)
- Qatlamlarning umumiy soni: 121

<p align="center">
 <img src="https://github.com/MisterFoziljon/Mashhur-Arxitekturalar/blob/main/rasmlar/DenseNet.jpg" width="800"/>
</p>

```DenseNet.ipynb``` notebookda arxitektura oxford_flowers102 dataseti uchun moslangan va noldan qurilgan. Train va Test qilingan.
