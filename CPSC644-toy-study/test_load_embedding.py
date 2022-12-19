from glob import glob

import numpy as np

files = sorted(glob('./results/embeddings/mnist-NA-val_acc_70%/*'))
batch_0_file = np.load(files[0])

image = batch_0_file['image']
label = batch_0_file['label_true']
embeddings = batch_0_file['embedding']

print(image.shape)
print(label.shape)
print(embeddings.shape)
