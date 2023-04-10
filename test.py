from test_helper import *

for i in range(20,30):
	probe(torch.tensor(poses[i]), i)

