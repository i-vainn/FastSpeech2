import torch

from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    num_mels = 80
    max_wav_value = 32768.0
    sampling_rate = 22050
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    mel_fmin = 0.0
    mel_fmax = 8000.0

@dataclass
class FastSpeechConfig:
    vocab_size = 300
    max_seq_len = 3000

    encoder_dim = 256
    encoder_n_layer = 4
    encoder_head = 2
    encoder_conv1d_filter_size = 1024

    decoder_dim = 256
    decoder_n_layer = 4
    decoder_head = 2
    decoder_conv1d_filter_size = 1024

    n_bins = 256
    min_energy = 0
    max_energy = 1
    min_pitch = 0
    max_pitch = 1

    fft_conv1d_kernel = (9, 1)
    fft_conv1d_padding = (4, 0)

    duration_predictor_filter_size = 256
    duration_predictor_kernel_size = 3
    dropout = 0.1
    
    PAD = 0
    UNK = 1
    BOS = 2
    EOS = 3

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'


@dataclass
class TrainConfig:
    checkpoint_path = "./model_new"
    logger_path = "./logger"
    mel_ground_truth = "./mels"
    alignment_path = "./alignments"
    data_path = './data/train.txt'
    wav_path = './data/LJSpeech-1.1/wavs'
    
    wandb_project = 'fastspeech2'
    
    text_cleaners = ['english_cleaners']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 2000
    n_warm_up_step = 4000

    learning_rate = 1e-3
    weight_decay = 1e-6
    grad_clip_thresh = 1.0
    decay_step = [500000, 1000000, 2000000]

    save_step = 3000
    log_step = 5
    clear_Time = 20

    batch_expand_size = 32
    

mel_config = MelSpectrogramConfig()
model_config = FastSpeechConfig()
train_config = TrainConfig()
stft = lambda wav: torch.stft(
    wav,
    mel_config.filter_length,
    mel_config.hop_length,
    mel_config.win_length,
    return_complex=False,
)