import time
import torch
import sys
import subprocess
import GPUtil
import os


def i_need_cards(num):
    device_ids = GPUtil.getAvailable(order='memory', limit=num,
                                     maxLoad=0.1, maxMemory=0.1, includeNan=False, excludeID=[], excludeUUID=[])

    card_ids = ",".join([str(id_) for id_ in device_ids])

    os.environ["CUDA_VISIBLE_DEVICES"] = card_ids

    print("Run on GPU", card_ids)

argslist = list(sys.argv)[1:]
num_gpus = torch.cuda.device_count()
i_need_cards(num_gpus)
argslist.append('--n_gpus={}'.format(num_gpus))
workers = []
job_id = time.strftime("%Y_%m_%d-%H%M%S")
argslist.append("--group_name=group_{}".format(job_id))

for i in range(num_gpus):
    argslist.append('--rank={}'.format(i))
    print(argslist)
    p = subprocess.Popen([str(sys.executable)]+argslist)
    workers.append(p)
    argslist = argslist[:-1]

for p in workers:
    p.wait()