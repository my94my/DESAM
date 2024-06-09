from argparse import ArgumentParser
from os import listdir, path, makedirs
from torch import Generator
from EEG_Diffusion import EEGDiffusion

generator = Generator(device='cuda')
generator.manual_seed(114514)


overlap_secs = 1
sample_rate = 250
overlap_samples = overlap_secs * sample_rate

chan_name = ('C3', 'Cz', 'C4')
for chan in chan_name:
    model_path = f'dataset/2B_{chan}/model'
    for sub in range(1, 10):
        image_path = f'../LMDA/BCIIV2b_3chan_tf/B0{sub}T/{chan}'
        save_path = f'../LMDA/BCIIV2b_3chan_tf_gen/B0{sub}G01/{chan}'
        makedirs(save_path, exist_ok=True)
        diffusion = EEGDiffusion(model_path=model_path)
        for i in listdir(image_path):
            img = path.join(image_path, i)
            new_img = diffusion.generate_spectrogram_from_spectrogram(
                    eeg_image_file=img,
                    eeg_fragment_length=4.5,
                    start_step=100,
                    steps=200,
                    mask_start_secs=overlap_secs,
                    generator=generator)
            new_img.save(path.join(save_path, i), "PNG")