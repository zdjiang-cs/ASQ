Requirements
---
Pytorch v1.6 <br>
CUDA v10.0 <br>
cuDNN v7.5.0 <br>

Organization of the code
---
* `client_module` is designed for each embedded device, including implementations of local training, gradient sparsification and quantification.<br>
* `communication_module` contains implementations of establishing communication links between the server and devices.<br>
* `training_module` contains implementations of models, datasets and data partition.<br>
* `server.py` is designed for the server, including implementations of model aggregation and model distribution. <br>

Usage
---
We conduct experiments on a hardware prototype system, which consists of an AMAX deep learning workstation as the server and 30 NVIDIA Jetson TX2 devices.<br>
To reproduce experiments, please migrate `client_module` to embedded devices (e.g., NVIDIA Jetson TX2), and run the following command for the devices:<br>
```python
    python client.py --master_ip 'MASTER_IP'
```
Please replace an argument for `--master_ip` with your server IP and run the following command for the server:<br>
```python
    python server.py --epoch 500
```
