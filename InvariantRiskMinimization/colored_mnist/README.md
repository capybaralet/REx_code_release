# Code repository for Risk Extrapolation Colored MNIST experiments

to train model with V-REx:
```
python main.py
```

to train model with IRMv1:
```
python main.py --rex False
```

to train model with MM-REx:
```
python main.py --mse False
```

to train model with Risk Interpolation:
```
python main.py --mse False --penalty_weight 1.0
```

to obtain REx line from plot of test_acc vs waterfall_activation_epoch in figure 12:
```
python main_waterfall.py
```

to obtain IRMv1 line from plot of test_acc vs waterfall_activation_epoch in figure 12:
```
python main_waterfall.py --rex False
```

to obtain plot of loss vs epochs from figure 12 that shows when memorization starts:
```
python main.py --n_restarts 1 --penalty_anneal_iters -1 --penalty_weight 0.0 --l2_regularizer_weight 0.0 --steps 401
```

to obtain plot of loss vs epochs from figure 13 that shows memorization can completely minimize stability penalties:
```
python main.py --n_restarts 1 --penalty_anneal_iters -1 --penalty_weight 0.0 --l2_regularizer_weight 0.0 --steps 25000
```

To make Figure 2 and Figure 12 (middle):
```
python make_Figure2.py
```


To train V-REx-HD on CMNIST:
```
python REx-HD.py 
```

To train V-REx-HD on hetero-CMNIST:
```
python REx-HD.py --hetero=1 --digit_shift=1
```
