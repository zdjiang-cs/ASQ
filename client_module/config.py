from typing import List


class ClientAction:
    LOCAL_TRAINING = "local_training"


class ServerAction:
    LOCAL_TRAINING = "local_training"


class ClientConfig:
    def __init__(self,
                 idx: int,
                 master_ip_addr: str,
                 action: str,
                 custom: dict = dict()
                 ):
        self.idx = idx
        self.master_ip_addr = master_ip_addr
        self.action = action
        self.custom = custom
        self.epoch_num: int = 1
        self.model = list()
        self.para = dict()
        self.resource = {"CPU": "1"}
        self.acc: float = 0
        self.loss: float = 1
        self.download_time: int = 0
        self.upload_time: int = 0
        self.train_time: int = 0
        self.compression_ratio = 0.5
        self.quantization_level = 8