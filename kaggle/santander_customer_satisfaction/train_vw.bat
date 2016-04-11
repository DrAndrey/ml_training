rem vw -d temp\scs.input_train -c -k --cache_file temp\scs.cache --passes 1 -f temp\scs.model --loss_function=logistic --binary
rem vw temp\scs.input_train -f temp\scs.model --binary --passes 20 -c -k --cache_file temp\scs.cache -q ff --adaptive --normalized --l1 0.00000001 --l2 0.0000001 -b 24
vw -d temp\scs.input_train -f temp\scs.model --loss_function logistic --readable_model a.txt