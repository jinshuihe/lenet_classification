import numpy as np

trainval_file = './images-cut/obstacle_trainval.txt'
train_file    = './images-cut/obstacle_train.txt'
val_file      = './images-cut/obstacle_val.txt'

idx = []
with open(trainval_file) as f:
  for line in f:
    idx.append(line.strip())
f.close()

idx = np.random.permutation(idx)

train_idx = sorted(idx[:len(idx)*11/12])
val_idx = sorted(idx[len(idx)*11/12:])

with open(train_file, 'w') as f:
  for i in train_idx:
    f.write('{}\n'.format(i))
f.close()

with open(val_file, 'w') as f:
  for i in val_idx:
    f.write('{}\n'.format(i))
f.close()

print ('Trainining set is saved to ' + train_file)
print ('Validation set is saved to ' + val_file)
