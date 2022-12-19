'''
Compute local curvature using diffusion curvature for input embeddings

'''
import numpy as np
import pandas as pd

from diffusion_curvature.core import DiffusionMatrix
from diffusion_curvature.laziness import curvature



def comp_curvature(X, labels, n_classes, data_df, k=20, diffusion_powers=8):
    class_stats = pd.DataFrame(columns = ['mean', 'std', 'class']) # mean, std, 

    # Diffusion Matrix
    P = DiffusionMatrix(X, kernel_type="adaptive anisotropic", k=k)

    # Curvature
    diffusion_curvatures = curvature(P, diffusion_powers=diffusion_powers)
    # data_df['curvature'] = diffusion_curvatures

    # Compute curvature related stats per class
    for ci in n_classes:
        c_rows = data_df.loc[data_df['class'] == ci]
        curvs = c_rows['curvature']

        # Append class stats row
        #class_stats.loc[len(class_stats.index)] = [np.mean(curvs), np.std(curvs), ci]
        class_stats = class_stats.append({'mean':np.mean(curvs), 'std':np.std(curvs), 'class': ci}, ignore_index=True)


    return diffusion_curvatures, class_stats


if __name__ == '__main__':
    X = np.random.rand(100,50) # N x dim

    curvature, class_stats = comp_curvature(X, [], [], [])
    data_df['curvature'] = curvature
    print(np.mean(curvature))