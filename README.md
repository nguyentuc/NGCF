# Neural Graph Collaborative Filtering
This is my PyTorch implementation for the paper:

> Neural Graph Collaborative Filtering. SIGIR 2019.


## Environment Requirement
The code has been run and tested under Python 3.8.13 and pytorch == 1.4.0


## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the parser function in NGCF/utility/parser.py).
* Gowalla dataset
```
python main.py --dataset gowalla --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0001 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 400 --verbose 1 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```

```
Best Iter=[38]@[32904.5]	recall=[0.15571	0.21793	0.26385	0.30103	0.33170], precision=[0.04763	0.03370	0.02744	0.02359	0.02088], hit=[0.53996	0.64559	0.70464	0.74546	0.77406], ndcg=[0.22752	0.26555	0.29044	0.30926	0.32406]
```


* Amazon-book dataset
```
python main.py --dataset amazon-book --regs [1e-5] --embed_size 64 --layer_size [64,64,64] --lr 0.0005 --save_flag 1 --pretrain 0 --batch_size 1024 --epoch 200 --verbose 50 --node_dropout [0.1] --mess_dropout [0.1,0.1,0.1]
```
## Supplement

* The parameter `negative_slope` of LeakyReLu was set to 0.2, since the default value of PyTorch and TensorFlow is different.
* If the arguement `node_dropout_flag` is set to 1, it will lead to higher calculational cost.