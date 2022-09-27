#!/bin/bash
python scripts/data_preparation/METR-LA/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/METR-LA/generate_training_data.py --history_seq_len 2016
python scripts/data_preparation/PEMS-BAY/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS-BAY/generate_training_data.py --history_seq_len 2016
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 12
python scripts/data_preparation/PEMS04/generate_training_data.py --history_seq_len 4032

