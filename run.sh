#!/bin/bash
#<network filename> <training filename> <test filename> <load weight(Load/New)> <task(Train/Predict)>
python SingleRes-Hybrid-BN.py Data/Train.h5 Data/TestGMMGC950_icdar2019.h5 Load Predict #For test
python evaluate_CER_from_prediction.py

