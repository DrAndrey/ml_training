#!/usr/bin/env bash

vw train_titanic.vw -f model.vw --binary --passes 20 -c -q ff --adaptive --normalized --l1 0.00000001 --l2 0.0000001 -b 24