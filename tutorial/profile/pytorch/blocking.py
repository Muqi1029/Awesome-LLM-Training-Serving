from time import perf_counter

import torch

a = torch.randn((200, 100), pin_memory=True)

tic_1 = perf_counter()
a.cuda()
print(f"Blocking time: {(perf_counter() - tic_1):.6f}")


tic_2 = perf_counter()
a.cuda(non_blocking=True)
print(f"Non-Blocking time: {(perf_counter() - tic_2):.6f}")
