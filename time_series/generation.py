from argparse import ArgumentParser
from torch import from_numpy, ones, normal, no_grad
from torch.cuda import Event, synchronize
from torch import load as load_model
from numpy import load as load_npy
from numpy import split, array, concatenate, save, sqrt
from imputers.SSSDS4Imputer import SSSDS4Imputer
from utils.util import calc_diffusion_hyperparams
from random import sample
from utils.util import sampling
from os import makedirs, environ


def sampling1(model, size, diffusion_hyperparams, cond, mask, only_generate_missing=True):
    """
    Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)

    Parameters:
    net (torch network):            the wavenet model
    size (tuple):                   size of tensor to be generated,
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated audio(s) in torch.tensor, shape=size
    """

    T = diffusion_hyperparams["T"]
    alpha = diffusion_hyperparams["Alpha"]
    alpha_bar = diffusion_hyperparams["Alpha_bar"]
    sigma = diffusion_hyperparams["Sigma"]
    assert len(alpha) == T
    assert len(alpha_bar) == T
    assert len(sigma) == T
    assert len(size) == 3

    print('begin sampling, total number of reverse steps = %s' % T)

    x = normal(0, 1, size=size).cuda()

    with no_grad():
        for t in range(T - 1, -1, -1):
            if only_generate_missing is True:
                x = x * (1 - mask).float() + cond * mask.float()
            diffusion_steps = (t * ones((size[0], 1))).cuda()  # use the corresponding reverse step
            epsilon_theta = model((x, cond, mask, diffusion_steps,))  # predict \epsilon according to \epsilon_\theta
            # update x_{t-1} to \mu_\theta(x_t)
            x = (x - (1 - alpha[t]) / sqrt(1 - alpha_bar[t]) * epsilon_theta) / sqrt(alpha[t])
            if t > 0:
                x = x + sigma[t] * normal(0, 1, size=size).cuda()  # add the variance term to x_{t-1}
    return x


def generate(data_file,
             batch_size,
             num_samples,
             save_path,
             testno,
             model_epoch,
             diffusion_hyperparams):
    # makedirs(save_path, exist_ok=True)

    # map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_hyperparams)
    for param in diffusion_hyperparams:
        if param != "T":
            diffusion_hyperparams[param] = diffusion_hyperparams[param].cuda()

    # predefine model
    model = SSSDS4Imputer(**model_config).cuda()

    # load checkpoint
    model.load_state_dict(load_model(f'results/test{testno}/{model_epoch}.pkl', map_location='cpu'))
    print('Successfully loaded model')

    ### Custom data loading and reshaping ###
    data = load_npy(data_file)
    print(f'Data shape: {data.shape}')
    n_batch = len(data) // batch_size
    data = split(data, n_batch)
    data = array(data)
    data = from_numpy(data).float().cuda()

    data_channels, data_length = data.shape[2:]
    mask_shape = data.shape[1:]
    mask_sections = range(250, 1125, 5)

    generated_eeg = []
    masks = []
    for batch in data:
        mask = ones(mask_shape)
        for r in sample(mask_sections, 3):
            mask[:, :, r:r + 5] = 0
        mask = mask.float().cuda()

        start = Event(enable_timing=True)
        end = Event(enable_timing=True)

        start.record()
        print(mask.shape, batch.shape)

        res = sampling(model, (batch_size, data_channels, data_length),
                       diffusion_hyperparams,
                       cond=batch, mask=mask)

        end.record()
        synchronize()

        print(f'Generated {num_samples} utterances of random_digit in {start.elapsed_time(end) // 1000} seconds')

        generated_eeg.append(res.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())

    generated_eeg = concatenate(generated_eeg, axis=0)
    save(f'{save_path}/generated_eeg.npy', generated_eeg)

    masks = concatenate(masks, axis=0)
    save(f'{save_path}/masks.npy', masks)

    print('Saving generated samples')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=1)
    parser.add_argument('--testno', type=int)
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=144)
    parser.add_argument('--data', type=str, default='./2a_train_3chan.npy')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    # environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    model_config = {
        "in_channels": 3,
        "out_channels": 3,
        "num_res_layers": 10,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 2000,
        "s4_d_state": 64,
        "s4_dropout": 0.0,
        "s4_bidirectional": 1,
        "s4_layernorm": 1
    }
    diffusion_hyperparams = {
        "T": 200,
        "beta_0": 0.0001,
        "beta_T": 0.02
    }
    generate(data_file=args.data,
             batch_size=args.batch_size,
             num_samples=args.samples,
             save_path='.',
             testno=args.testno,
             model_epoch=args.epoch,
             diffusion_hyperparams=diffusion_hyperparams)
