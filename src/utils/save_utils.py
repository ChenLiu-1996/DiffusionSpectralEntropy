import os
import numpy as np
import torch
from attribute_hashmap import AttributeHashmap

def save_numpy(config: AttributeHashmap, batch_idx: int, numpy_filename: str,
               image_batch: torch.Tensor, label_true_batch: torch.Tensor,
               embedding_batch: torch.Tensor, np_dtype: np.dtype = np.float16):

    image_batch = image_batch.cpu().detach().numpy().astype(np_dtype)
    label_true_batch = label_true_batch.cpu().detach().numpy().astype(np_dtype)
    embedding_batch = embedding_batch.cpu().detach().numpy().astype(np_dtype)
    # channel-first to channel-last
    image_batch = np.moveaxis(image_batch, 1, -1)

    # Save the images, labels, and predictions as numpy files for future reference.
    save_path_numpy = '%s/embeddings/%s/' % (config.output_save_path,
                                             numpy_filename)
    os.makedirs(save_path_numpy, exist_ok=True)

    with open(
            '%s/%s' %
        (save_path_numpy, 'batch_%s.npz' % str(batch_idx).zfill(5)),
            'wb+') as f:
        np.savez(f,
                 image=image_batch,
                 label_true=label_true_batch,
                 embedding=embedding_batch)
    return
