import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt

sys.path.insert(0, os.path.abspath('../'))
from stylegan2_model import StyleGAN2Model


def run_stylegan2(domain: str, num_samples: int = 8) -> None:
    print('\n\ndomain: ', domain)

    noise_list, output_init_list, output_pretrained_list = [], [], []
    stylegan2 = StyleGAN2Model(
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        domain=domain)
    stylegan2.eval()

    for random_seed in range(num_samples):
        noise = torch.from_numpy(
            np.random.RandomState(random_seed).randn(
                1, stylegan2.model.z_dim)).to(stylegan2.device)
        noise_list.append(noise)

    with torch.no_grad():
        for noise in noise_list:
            output_init = stylegan2.forward(noise)
            # [B, C, H, W] to [H, W, C]
            output_init = \
                output_init.permute(0, 2, 3, 1)[0].cpu().numpy()
            output_init_list.append(output_init)

    stylegan2.restore_model()
    with torch.no_grad():
        for noise in noise_list:
            output_pretrained = stylegan2.forward(noise)
            # [B, C, H, W] to [H, W, C]
            output_pretrained = \
                output_pretrained.permute(0, 2, 3, 1)[0].cpu().numpy()
            output_pretrained_list.append(output_pretrained)

    # Plot the results.
    plt.rcParams['figure.figsize'] = [5, 2 * num_samples]
    fig = plt.figure()
    for i in range(num_samples):
        noise = noise_list[i].cpu().detach().numpy()
        output_init = output_init_list[i]
        output_pretrained = output_pretrained_list[i]
        ax = fig.add_subplot(num_samples, 3, 1 + i * 3)
        ax.imshow(noise)
        ax.set_title('Input\n%s' % str(noise.shape))
        ax = fig.add_subplot(num_samples, 3, 2 + i * 3)
        ax.imshow(output_init)
        ax.set_title('Output\n[initial weights]\n%s' % str(output_init.shape))
        ax = fig.add_subplot(num_samples, 3, 3 + i * 3)
        ax.imshow(output_pretrained)
        ax.set_title('Output\n[pretrained weights]\n%s' %
                     str(output_pretrained.shape))

    plt.tight_layout()
    os.makedirs('./unit_test_output/', exist_ok=True)
    fig.savefig('./unit_test_output/stylegan2_%s.png' % domain)

    print('\n\nFetched Latent Features:')
    latent_outputs = stylegan2.fetch_latent()
    for key in latent_outputs.keys():
        print(key, latent_outputs[key].shape)


if __name__ == '__main__':
    for domain in ['ffhq_1024']:
        run_stylegan2(domain)
