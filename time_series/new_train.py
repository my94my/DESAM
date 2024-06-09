from argparse import ArgumentParser
from random import sample
from numpy import split, array
from numpy import load as load_npy
from torch import from_numpy, save, ones
from torch.optim import Adam
from torch.nn import MSELoss
from imputers.SSSDS4Imputer import SSSDS4Imputer
from utils.util import calc_diffusion_hyperparams, training_loss
from os import makedirs


def train(data_file,
          epoch,
          epoch_save,
          learning_rate,
          batch_size,
          model_config,
          diffusion_hyperparams,
          testno):
    """
    Train Diffusion Models

    Parameters:
    output_directory (str):         save model checkpoints to this path
    ckpt_iter (int or 'max'):       the pretrained checkpoint to be loaded; 
                                    automatically selects the maximum iteration if 'max' is selected
    data_path (str):                path to dataset, numpy array.
    n_iters (int):                  number of iterations to train
    iters_per_ckpt (int):           number of iterations to save checkpoint, 
                                    default is 10k, for models with residual_channel=64 this number can be larger
    iters_per_logging (int):        number of iterations to save training log and compute validation loss, default is 100
    learning_rate (float):          learning rate

    use_model (int):                0:DiffWave. 1:SSSDSA. 2:SSSDS4.
    only_generate_missing (int):    0:all sample diffusion.  1:only apply diffusion to missing portions of the signal
    masking(str):                   'mnr': missing not at random, 'bm': blackout missing, 'rm': random missing
    missing_k (int):                k missing time steps for each feature across the sample length.
    """

    # generate experiment (local) path
    save_path = f'results/test{testno}'
    if testno > 0:
        makedirs(save_path)
    # map diffusion hyperparameters to gpu
    diffusion_hyperparams = calc_diffusion_hyperparams(**diffusion_hyperparams)
    for param in diffusion_hyperparams:
        if param != "T":
            diffusion_hyperparams[param] = diffusion_hyperparams[param].cuda()

    # predefine model
    model = SSSDS4Imputer(**model_config).cuda()

    # define optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # load checkpoint
    # if ckpt_iter == 'max':
    #     ckpt_iter = find_max_epoch(output_directory)
    # if ckpt_iter >= 0:
    #     try:
    #         # load checkpoint file
    #         model_path = os.path.join(output_directory, '{}.pkl'.format(ckpt_iter))
    #         checkpoint = torch.load(model_path, map_location='cpu')
    #
    #         # feed model dict and optimizer state
    #         model.load_state_dict(checkpoint['model_state_dict'])
    #         if 'optimizer_state_dict' in checkpoint:
    #             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #
    #         print('Successfully loaded model at iteration {}'.format(ckpt_iter))
    #     except:
    #         ckpt_iter = -1
    #         print('No valid checkpoint model found, start training from initialization try.')
    # else:
    #     ckpt_iter = -1
    #     print('No valid checkpoint model found, start training from initialization.')

    ### Custom data loading and reshaping ###

    data = load_npy(data_file)
    print(f'Data shape: {data.shape}')
    n_batch = len(data) // batch_size
    data = split(data, n_batch)
    data = array(data)
    data = from_numpy(data).float().cuda()

    mask_shape = data.shape[1:]
    mask_sections = range(50, 300, 5)

    # training
    i = 1
    while i < epoch + 1:
        loss_sum = 0
        mask = ones(mask_shape)
        for r in sample(mask_sections, 5):
            mask[:, :, r:r + 5] = 0
        loss_mask = ~mask.bool()
        mask = mask.float().cuda()
        for batch in data:
            # back-propagation
            optimizer.zero_grad()
            X = batch, batch, mask, loss_mask
            loss = training_loss(model, MSELoss(), X, diffusion_hyperparams)
            loss_sum += loss.item()
            loss.backward()
            optimizer.step()

        res = f'Epoch: {i}\tloss: {loss_sum / n_batch}\n'
        print(res)
        with open(f'{save_path}/loss.txt', 'a') as file:
            file.write(res)

        # save checkpoint
        if not i % epoch_save:
            print(f'Saving model at epoch {i}')
            save(model.state_dict(), f"{save_path}/{i}.pkl")

        i += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--subject', type=int, default=3)
    parser.add_argument('--testno', type=int, default=0)
    args = parser.parse_args()

    model_config = {
        "in_channels": 3,
        "out_channels": 3,
        "num_res_layers": 8,
        "res_channels": 256,
        "skip_channels": 256,
        "diffusion_step_embed_dim_in": 128,
        "diffusion_step_embed_dim_mid": 512,
        "diffusion_step_embed_dim_out": 512,
        "s4_lmax": 125,
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
    train(data_file=f"BCIIV1_3chan/A0{args.subject}E_data.npy",
          epoch=2000,
          epoch_save=10,
          learning_rate=2e-4,
          batch_size=20,
          model_config=model_config,
          diffusion_hyperparams=diffusion_hyperparams,
          testno=args.testno)
