1. [Быстрая реализация Dot Plot DNA на Python](https://gitlab.com/transposable_elements/src/-/blob/main/utils/fast_dot_plot_dna.py)
2. Генерация датасета
   - [Генерация графов-графиков](https://gitlab.com/transposable_elements/src/-/blob/main/generate_dataset/generate.py)
   - [Формирование файлов csv](https://gitlab.com/transposable_elements/src/-/blob/main/generate_dataset/write_csv.py)

| ![](images/LTR.png) LTR (повторы на концах) | ![](images/TIR.png) TIR (палиндромы на концах) |
|------------------------------|------------------------------|
| ![](images/NO.png) NO (нет повторов) | ![](images/INNER.png) INNER (повторы внутри) |


3. Результаты обучения GNN

![](images/LTR_NO_15EPOCHS.png) LTR-NO

![](images/LTR_TIR_15EPOCHS.png) LTR-TIR

![](images/LTR_INNER_50EPOCHS.png) LTR-INNER


[Ссылка на данные](https://drive.google.com/drive/folders/1JHspYMC_GHS-FgYl7MKfZsH-PzcG2EXR?usp=sharing)