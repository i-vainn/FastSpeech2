import torch
from torch import nn
from .configs import FastSpeechConfig


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class DurationPredictor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(DurationPredictor, self).__init__()

        self.input_size = model_config.encoder_dim
        self.filter_size = model_config.duration_predictor_filter_size
        self.kernel = model_config.duration_predictor_kernel_size
        self.conv_output_size = model_config.duration_predictor_filter_size
        self.dropout = model_config.dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, model_config):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = DurationPredictor(model_config)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = nn.functional.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)

        if target is not None:
            output = self.LR(x, target, mel_max_length=mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = torch.exp(self.duration_predictor(x)) - 1
            duration_predictor_output = (
                (duration_predictor_output + 0.5) * alpha).int()
            output = self.LR(x, duration_predictor_output)
            mel_pos = torch.stack(
                [torch.Tensor([i+1 for i in range(output.size(1))])]).long().to(output.device)

            return output, mel_pos


class VarianceAdaptor(nn.Module):
    """ Duration Predictor """

    def __init__(self, model_config: FastSpeechConfig):
        super(VarianceAdaptor, self).__init__()

        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = DurationPredictor(model_config)
        self.energy_predictor = DurationPredictor(model_config)

        self.speaker_embeds = nn.Embedding(50, model_config.encoder_dim)
        self.energy_embeds = nn.Embedding(model_config.n_bins, model_config.encoder_dim)
        self.pitch_embeds = nn.Embedding(model_config.n_bins, model_config.encoder_dim)
        self.energy_bins = nn.Parameter(
            torch.linspace(model_config.min_energy, model_config.max_energy, model_config.n_bins - 1),
            requires_grad=False,
        )
        self.pitch_bins = nn.Parameter(
            torch.linspace(model_config.min_pitch, model_config.max_pitch, model_config.n_bins - 1),
            requires_grad=False,
        )

    def embed_pitch(self, x, target):
        pred = self.pitch_predictor(x)
        if target is not None:
            embedding = self.pitch_embeds(torch.bucketize(target, self.pitch_bins))
        else:
            embedding = self.pitch_embeds(torch.bucketize(pred, self.pitch_bins))
        return pred, embedding

    def embed_energy(self, x, target):
        pred = self.energy_predictor(x)
        if target is not None:
            embedding = self.energy_embeds(torch.bucketize(target, self.energy_bins))
        else:
            embedding = self.energy_embeds(torch.bucketize(pred, self.energy_bins))
        return pred, embedding

    def forward(
        self,
        encoder_output,
        speaker_id,
        mel_max_length=None,
        length_target=None,
        pitch_target=None,
        energy_target=None,
        alpha=1.0,
    ):
        duration_prediction = None
        mel_pos = None
        if self.training:
            length_regulator_output, duration_prediction = self.length_regulator(encoder_output,
                                                                                target=length_target,
                                                                                alpha=alpha,
                                                                                mel_max_length=mel_max_length)
        else:
            length_regulator_output, mel_pos = self.length_regulator(encoder_output, alpha=alpha)
        
        energy_prediction, energy_embedding = self.embed_energy(
            length_regulator_output, energy_target
        )
        pitch_prediction, pitch_embedding = self.embed_pitch(
            length_regulator_output, pitch_target
        )
        speaker_embed = self.speaker_embeds(speaker_id).unsqueeze(1).expand(pitch_embedding.size())

        length_regulator_output = length_regulator_output + pitch_embedding + energy_embedding + speaker_embed
        
        return (
            length_regulator_output,
            mel_pos,
            duration_prediction,
            energy_prediction,
            pitch_prediction,
        )