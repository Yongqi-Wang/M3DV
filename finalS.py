# -*- coding: utf-8 -*-
"""

@author: dragon
"""

import numpy as np
import os
import dense
import csv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'

# test data
test_data_all = os.listdir('./test')
test_data_all.sort()
test_data = np.ones((117, 32, 32, 32, 1))
filename = [0 for _ in range(117)] 

m = 0
for data in test_data_all:
    filename[m] = int(data[9:len(data)-4])
    m = m+1
filename.sort()

for i in range(0, 117):
     npz = np.load(os.path.join('./test', 'candidate'+str(filename[i])+'.npz'))
     temp =npz['voxel']*npz['seg']
     test_data[i, :, :, :, 0] = temp[34:66, 34:66, 34:66]

# compile and test
model1 = dense.get_compiled()
model2 = dense.get_compiled()
model3 = dense.get_compiled()

model1.load_weights('./models/ep1.h5')
model2.load_weights('./models/ep2.h5')
model3.load_weights('./models/ep3.h5')

pre_res = model1.predict(test_data) + model2.predict(test_data) + model3.predict(test_data)

# result writer
res_writer = open('./result.csv', 'w', encoding='utf-8', newline='')
csv_w = csv.writer(res_writer)
csv_w.writerow(["Id", "Predicted"])

for i in range(0, 117):
    csv_w.writerow(['candidate'+str(filename[i]), pre_res[i, 1]])
res_writer.close()
