
# Data Preparation

## Download Raw Data

You can download all the raw datasets at [Google Drive](https://drive.google.com/file/d/1PY7IZ3SchpyXfNIXs71A2GEV29W5QCv2/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1CXLxeHxHIMWLy3IKGFUq8g?pwd=blf8), and unzip them to `datasets/raw_data/`.

## Pre-process Data

You can pre-process all data via:

```bash
cd /path/to/your/project
bash scripts/data_preparation/all.sh
```

Then the `dataset` directory will look like this:

```text
datasets
   ├─METR-LA
   ├─METR-BAY
   ├─PEMS04
   ├─raw_data
   |    ├─PEMS04
   |    ├─PEMS-BAY
   |    ├─METR-LA
   ├─README.md
```
