#!/usr/bin/env bash

vw -d temp/scs.input_train -f temp/scs.model --binary -q ff --adaptive --normalized --l1 0.00000001 --l2 0.0000001 -b 24
