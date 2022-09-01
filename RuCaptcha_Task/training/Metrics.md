---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
from grcnn import metrics
from grcnn.utils import find_best_model_epoch
```

```python
dataset_dir_path = "/srv/data_science/storage/price_ocr"
max_text_len = 17
width, height = 64, 32
n_classes = 10
df_name = f'test_set_len{max_text_len}'
# df_name = 'train_set_relabel_len17'
# df_name = 'bad_samples'
# df_name = "zeros_test"
# df_name = "len1_test"
# df_name = f'train_set_len{max_text_len}'
# df_name = f'nov_2019_markets37_val_set_len{max_text_len}'
model_num = 33
epoch = find_best_model_epoch(model_num)
# epoch = 23 # m14
# epoch = 11 # m16
# epoch = 28 # m15
# epoch = 60 # m4
# epoch = 62  # m2
# batch_size = 1024
batch_size = 20
image_dir = f"images_train_size{width}x{height}"
# image_dir = f"images_train_padding_zero_size{width}x{height}"
# image_dir = 'nov_2019_markets37_processed_images'
accuracy_threshold = 99.9
grcl_fsize = 3
# grcl_niter = 3
grcl_niter = 5
lstm_units = 512
debug = 0
dist_filter = lambda x: x > 0
# text_filter = lambda x: '6' in x
text_filter = None
```

```python
metrics.test_nn(dataset_dir_path, df_name, image_dir, model_num, epoch, batch_size, max_text_len, height, width, n_classes, 
                grcl_fsize, grcl_niter, lstm_units, debug, dist_filter, text_filter)
```

```python
metrics.test_metric_by_markets(dataset_dir_path, df_name, image_dir, model_num, epoch, batch_size, max_text_len, height, width, accuracy_threshold)
```

```python
d = {'europa-ts.ru': (100.0, 0.013554, 99.618), 'okmarket.ru': (100.0, 0.000962, 99.8), 'dixy.ru': (100.0, 0.0081, 99.496), 'spar.ru': (100.0, 0.005454, 99.065), '5ka.ru': (100.0, 0.006336, 99.284), 'lenta.com': (100.0, 0.012511, 99.58), 'magnit-info.ru': (100.0, 0.000722, 99.783), 'auchan.ru': (100.0, 0.0, 100.0), 'globus.ru': (100.0, 0.00818, 99.18), 'perekrestok.ru': (100.0, 0.023418, 99.211), 'megamart.ru': (100.0, 0.026626, 99.127), 'billa.ru': (100.0, 0.001283, 99.776), 'verno-info.ru': (100.0, 0.000722, 99.73), 'mgnl.ru': (100.0, 0.0, 100.0), 'krasnoeibeloe.ru': (100.0, 0.014596, 98.847)}
```

```python
for k, v in d.items():
    print(k)
    print(v)
```

```python
# d = {2: (4604, 5073), 3: (1363, 1568), 1: (12, 15), 4: (117, 134), 5: (2, 4)} # m2e62
# d = {2: (5008, 5073), 3: (1422, 1568), 1: (13, 15), 4: (117, 134), 5: (4, 4)} # m4e60; better by 7%
# d = {2: (4624, 5073), 3: (1382, 1568), 1: (10, 15), 4: (112, 134), 5: (4, 4)} # m14e23
# d = {2: (4637, 5073), 3: (1348, 1568), 4: (116, 134), 5: (3, 4), 1: (12, 15)} # m23
# d = {2: (4085, 5073), 3: (1360, 1568), 1: (12, 15), 4: (111, 134), 5: (4, 4)}
d = {2: (5053, 5073), 3: (1563, 1568), 1: (12, 15), 4: (130, 134), 5: (4, 4)} # m33
```

```python
total = 0
true = 0
for k, (t1, t2) in d.items():
    true += t1
    total += t2
print(true, total, total - true, round(true / total * 100, 3))
```

```python
def print_metrics(test_d, train_d):
    for k in train_d.keys():
        if k in test_d:
            print(k, test_d[k], train_d[k])
```

```python
dataset_dir_path = '/srv/data_science/storage/product_code_ocr'
df_name = 'test_set_google_bs_14'
metrics.test_google(dataset_dir_path, df_name)
```

```python
total = 0
true = 0
for k, (t1, t2) in d.items():
    true += t1
    total += t2
print(true, total, total - true, true / total)
```

```python
d1 = {'dilan.ru': (100.0, 0.037242, 23.093), 'bonus': (100.0, 0.043281, 23.324), 'kirmarket.ru': (100.0, 0.037242, 84.286), 'bristol.ru': (100.0, 0.058928, 49.81), 'spar.ru': (100.0, 0.052706, 50.935), 'somelie.ru': (100.0, 0.032941, 37.908), 'raz dva': (100.0, 0.040353, 30.711), 'globus.ru': (100.0, 0.044105, 66.516), 'krasnoeibeloe.ru': (100.0, 0.058379, 53.318), 'okmarket.ru': (100.0, 0.043098, 45.455), 'magnit-info.ru': (100.0, 0.04319, 57.829), 'samberi.com': (100.0, 0.045203, 59.284), 'verno-info.ru': (100.0, 0.037242, 82.045), 'test.ru': (100.0, 0.045569, 47.081), 'europa-ts.ru': (100.0, 0.037242, 89.709), 'lenta.com': (100.0, 0.040902, 69.045), 'vinovodochnyj': (100.0, 0.04319, 31.385), 'stolica': (100.0, 0.04319, 30.303), 'петромост.рф': (100.0, 0.055909, 62.931), 'perekrestok.ru': (100.0, 0.04319, 58.244), 'semya.ru': (100.0, 0.056549, 50.641), 'megamart.ru': (100.0, 0.049869, 40.932), 'ярче.рф': (100.0, 0.041909, 71.618), 'garant': (100.0, 0.040811, 80.178), '5ka.ru': (100.0, 0.044562, 71.345), '7-ya.ru': (100.0, 0.037242, 75.764), 'edelveis.ru': (100.0, 0.05966, 30.793), 'da.ru': (100.0, 0.059752, 52.62), 'gradusi.net': (100.0, 0.05719, 61.982), 'monetka.ru': (100.0, 0.053987, 57.564), 'auchan.ru': (100.0, 0.043007, 49.769), 'mgnl.ru': (100.0, 0.040719, 68.009), 'billa.ru': (100.0, 0.040353, 68.035), 'maxi-retail.ru': (100.0, 0.041085, 68.5), 'grinn-corp.ru': (100.0, 0.04319, 70.779), 'maria-ra.ru': (100.0, 0.043922, 58.811), 'dixy.ru': (100.0, 0.04383, 70.412)}
```

```python
d2 = {'europa-ts.ru': (100.0, 0.047714, 81.679), 'okmarket.ru': (100.0, 0.048275, 75.848), 'dixy.ru': (100.0, 0.043624, 81.612), 'spar.ru': (100.0, 0.043624, 75.935), '5ka.ru': (100.0, 0.043544, 83.294), 'lenta.com': (100.0, 0.043624, 66.106), 'magnit-info.ru': (100.0, 0.043624, 79.565), 'auchan.ru': (100.0, 0.043544, 64.691), 'globus.ru': (100.0, 0.048275, 74.59), 'perekrestok.ru': (100.0, 0.043544, 82.105), 'megamart.ru': (100.0, 0.043544, 55.313), 'billa.ru': (100.0, 0.047313, 78.3), 'verno-info.ru': (100.0, 0.043624, 79.245), 'mgnl.ru': (100.0, 0.043624, 81.37), 'krasnoeibeloe.ru': (100.0, 0.055172, 56.484)}
```

```python
threhsholds = [(k, d2[k][1] if k in d2 else d1[k][1]) for k, x in d1.items()]
```

```python
print("\n".join([f"{k}={v}"for k, v in threhsholds]))
```

```python

```
