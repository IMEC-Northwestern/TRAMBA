import librosa
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import math
import os
import soundfile as sf
from scipy.signal import decimate
from mamba_ssm import Mamba
import torch.nn as nn 
from auraloss.freq import STFTLoss, MultiResolutionSTFTLoss, apply_reduction
import torch.nn.functional as F
import shutil

def process_audio(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    frames = librosa.util.frame(audio, frame_length=8192, hop_length=4096).T
    reshaped_array = np.expand_dims(frames, axis=1)
    reshaped_tensor = torch.from_numpy(reshaped_array.copy())

    return reshaped_tensor

class TrainDataset(Dataset):
    def __init__(self, gs_folder_path, imu_folder_path):
        self.gs_file_paths = [os.path.join(gs_folder_path, f) for f in sorted(os.listdir(gs_folder_path)) if os.path.isfile(os.path.join(gs_folder_path, f))and f.lower().endswith('.wav')]
        self.imu_file_paths = [os.path.join(imu_folder_path, f) for f in sorted(os.listdir(imu_folder_path)) if os.path.isfile(os.path.join(imu_folder_path, f))and f.lower().endswith('.wav')]
    
    def __len__(self):
        return len(self.gs_file_paths)
    
    def __getitem__(self, idx):
        gs_file_path = self.gs_file_paths[idx]
        gs_frames = process_audio(gs_file_path)
        imu_file_path = self.imu_file_paths[idx]
        imu_frames = process_audio(imu_file_path)
        return gs_frames, imu_frames


class DevelDataset(Dataset):
    def __init__(self, gs_folder_path, imu_folder_path):
        self.gs_file_paths = [os.path.join(gs_folder_path, f) for f in sorted(os.listdir(gs_folder_path)) if os.path.isfile(os.path.join(gs_folder_path, f))and f.lower().endswith('.wav')]
        self.imu_file_paths = [os.path.join(imu_folder_path, f) for f in sorted(os.listdir(imu_folder_path)) if os.path.isfile(os.path.join(imu_folder_path, f))and f.lower().endswith('.wav')]
    
    def __len__(self):
        return len(self.gs_file_paths)
    
    def __getitem__(self, idx):
        gs_file_path = self.gs_file_paths[idx]
        gs_frames = process_audio(gs_file_path)
        imu_file_path = self.imu_file_paths[idx]
        imu_frames = process_audio(imu_file_path)
        return gs_frames, imu_frames
    
def collate_fn(batch):
    target = torch.cat([item[0] for item in batch], dim=0).float()
    data = torch.cat([item[1] for item in batch], dim=0).float()
    return [target, data]

class PixelShuffle1D(torch.nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle1D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)

        return x

def down_sampling(input_channel, output_channel, pad_size, k_size, s_size):
    conv_layer = nn.Conv1d(input_channel, output_channel, padding=pad_size, kernel_size=k_size, stride=s_size)

    conv = nn.Sequential(
        conv_layer,
        nn.LeakyReLU(0.2)
    )
    return conv

def up_sampling(input_channel, output_channel, pad_size, k_size, s_size):
    conv_layer = nn.Conv1d(input_channel, output_channel, padding=pad_size, kernel_size=k_size, stride=s_size)
    nn.init.orthogonal_(conv_layer.weight)
    conv = nn.Sequential(
        conv_layer,
        nn.Dropout1d(0.5),
        nn.LeakyReLU(0.2),
        PixelShuffle1D(4)
    )
    return conv

def out_layer(input_channel, output_channel, pad_size, k_size, s_size):
    conv = nn.Sequential(
        nn.Conv1d(input_channel, output_channel, padding=pad_size, kernel_size=k_size, stride=s_size),
        PixelShuffle1D(4)
    )
    return conv

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=1024, dropout=0.1, max_position_embeddings=64):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=dropout, max_len=max_position_embeddings)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.d_model = d_model
        self.sqrt_d_model = torch.sqrt(torch.tensor(d_model, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        src = src * self.sqrt_d_model
        src = self.pos_encoder(src)
        src = self.dropout(src)
        src = src.permute(1,0,2)
        output = self.transformer_encoder(src)
        return output

class SAFiLM(nn.Module):
    def __init__(self, n_step, block_size, n_filters):
        super(SAFiLM, self).__init__()
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step
        self.rnn = TransformerModel(d_model=n_filters,nhead=2,num_layers=3,max_position_embeddings=64)
        
    def make_normalizer(self, x_in):
        b, c, l = x_in.shape
        x_in_down = F.max_pool1d(x_in, kernel_size=self.block_size, stride=self.block_size, padding=0)
        x_in_down = x_in_down.permute(2,0,1)
        x_rnn = self.rnn(x_in_down)
        x_rnn = x_rnn.permute(0,2,1)
        return x_rnn

    def apply_normalizer(self, x_in, x_norm):
        b, c, l = x_in.shape
        n_blocks = l // self.block_size
        x_norm = x_norm.reshape(b, c, n_blocks, 1)
        x_in = x_in.view(b, c, n_blocks, self.block_size)
        x_out = x_norm * x_in
        x_out = x_out.reshape(b, c, n_blocks * self.block_size)
        return x_out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_filters, self.n_step)


    def forward(self, x):
        assert x.dim() == 3, 'Input should be tensor with dimension (batch_size, num_features, steps).'
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x

def bottleneck_layer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout):
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                               dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
    bottleneck = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
    return bottleneck

def mamba_bottleneck_layer(d_model, d_state, d_conv, expand):
    model = Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
    )
    return model

class UNet1d_64_3layer(nn.Module):
    def __init__(self):
        super(UNet1d_64_3layer, self).__init__( )
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # down sampling layer
        self.down_conv1 = down_sampling(1,64,pad_size=32,k_size=65, s_size=4)
        self.down_conv2 = down_sampling(64,128,8,17,4)
        self.down_conv3 = down_sampling(128,256,3,7,4)
        
        # bottleneck layer
        self.bottleneck = mamba_bottleneck_layer(d_model=256, d_state=16, d_conv=4, expand=2)
        
        # upsampling layer
        self.up_conv1 = up_sampling(256,512,3,7,1)
        self.up_conv2 = up_sampling(128,256,8,17,1)
        
        # output layer
        self.out = out_layer(64,4,32,65,1)
        
        # SAFiLM
        self.safilm1 = SAFiLM(n_step=2048, block_size=32, n_filters=64)
        self.safilm2 = SAFiLM(n_step=512, block_size=8, n_filters=128)

        self.safilm1u = SAFiLM(n_step=512, block_size=8, n_filters=128)
        self.safilm2u = SAFiLM(n_step=2048, block_size=32, n_filters=64)
        
    def forward(self, imu):
        
        imu_x1 = self.down_conv1(imu)
        imu_x1 = self.safilm1(imu_x1)
        
        imu_x2 = self.down_conv2(imu_x1)
        imu_x2 = self.safilm2(imu_x2)
        
        imu_x3 = self.down_conv3(imu_x2)
        imu_x3_p = imu_x3.permute(0,2,1)
        
        # bottleneck
        imu_b = self.bottleneck(imu_x3_p)
        imu_b_p = imu_b.permute(0,2,1)
        imu_b_c = imu_b_p + imu_x3
        
        # up sampling
        imu_u1 = self.up_conv1(imu_b_c)
        imu_u1 = self.safilm1u(imu_u1)
        imu_c1 = imu_u1 + imu_x2
        
        imu_u2 = self.up_conv2(imu_c1)
        imu_u2 = self.safilm2u(imu_u2)
        imu_c2 = imu_u2 + imu_x1
        
        imu_u3 = self.out(imu_c2)
        imu_u3 = torch.tanh(imu_u3)
        imu_c3 = imu_u3 + imu

        return imu_c3


def stft(x, fft_size, hop_size, win_length, window):
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window,return_complex=True)
    stft_view = torch.view_as_real(x_stft)
    real = stft_view[..., 0]
    imag = stft_view[..., 1]
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):

    def __init__(self):
        super(SpectralConvergengeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


class LogSTFTMagnitudeLoss(torch.nn.Module):

    def __init__(self):
        super(LogSTFTMagnitudeLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window"):
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(self, x, y):
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):

    def __init__(self,
                 fft_sizes=[1024, 2048, 512],
                 hop_sizes=[120, 240, 50],
                 win_lengths=[600, 1200, 240],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1):
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss

def print_lr(optimizer):
    for param_group in optimizer.param_groups:
        print("Current learning rate is: {}".format(param_group['lr']))


def save_checkpoint(state, filename):
    torch.save(state, filename)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet1d_model = UNet1d_64_3layer().to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(unet1d_model.parameters(), lr = 3e-4)
mrstftloss = MultiResolutionSTFTLoss(factor_sc=0.5,factor_mag=0.5).to(device)


def load_checkpoint(filepath, model, optimizer=None):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    return epoch

def main():

    train_ground_file_path = "Please enter your train data ground truth path"
    train_IMU_file_path = "Please enter your train data low res path"

    devel_ground_file_path = "Please enter your devel data ground truth path"
    devel_IMU_file_path = "Please enter your devel data low res path"

    train_dataset = TrainDataset(train_ground_file_path, train_IMU_file_path)
    devel_dataset = DevelDataset(devel_ground_file_path, devel_IMU_file_path)

    train_loader = DataLoader(dataset = train_dataset,
                            batch_size = 8,
                            shuffle = True,
                            collate_fn=collate_fn
                            )


    devel_loader = DataLoader(dataset = devel_dataset,
                            batch_size = 8,
                            shuffle = False,
                            collate_fn=collate_fn
                            )
    
    def train(epoch):
        unet1d_model.train()
        avg_loss = 0.0
        total_samples = 0
        for i, (train_reshaped_ground_tensor, train_reshaped_IMU_tensor) in enumerate(train_loader):
        
            ground_truth = train_reshaped_ground_tensor.to(device)
            y = torch.squeeze(ground_truth, axis=1)
        
            imu = train_reshaped_IMU_tensor.to(device)

            stft_output = unet1d_model(imu)
            x = torch.squeeze(stft_output, axis=1)
        
            sc_loss, mag_loss = mrstftloss(x,y)
            stft_loss = sc_loss + mag_loss + criterion(stft_output, ground_truth)
            avg_loss = (avg_loss * i + stft_loss.item())/(i+1)
            
            total_samples += 1

            optimizer.zero_grad()
            stft_loss.backward()
            torch.nn.utils.clip_grad_norm_(unet1d_model.parameters(), max_norm=1.0)
            optimizer.step()
        
        print(f'Epoch [{epoch+1}], avg_Loss: {avg_loss:.4f}')

    def evaluate(epoch):
        unet1d_model.eval()
        avg_loss = 0

        with torch.no_grad():
            for i, (devel_reshaped_ground_tensor, devel_reshaped_IMU_tensor) in enumerate(devel_loader):
           
                ground_truth = devel_reshaped_ground_tensor.to(device)
                y = torch.squeeze(ground_truth, axis=1)
                imu = devel_reshaped_IMU_tensor.to(device)
                stft_output = unet1d_model(imu)
                x = torch.squeeze(stft_output, axis=1)
                sc_loss, mag_loss = mrstftloss(x,y)
                stft_loss = sc_loss + mag_loss + criterion(stft_output, ground_truth)
                avg_loss = (avg_loss * i + stft_loss.item()) / (i + 1)
            print (f'Epoch [{epoch+1}], avg_loss = {avg_loss:.4f}')
        return avg_loss


    model_checkpoint_path = 'best_model.pth.tar (Please enter the pre-trained model path)'
    epoch_loaded = load_checkpoint(model_checkpoint_path, unet1d_model)
    print(f"Model loaded from epoch {epoch_loaded}")

    iters = 2000
    iters_to_copy = [100, 200, 300, 500, 1000, 1500, 2000]

    best_model_pth = 'best_model.pth (Please enter the path that you want to save the model)'
    best_model_tar = 'best_model.pth.tar (Please enter the path that you want to save the model)'

    for epoch in range(epoch_loaded, epoch_loaded + iters):
        train(epoch)  
        evaluate(epoch)

        torch.save(unet1d_model, best_model_pth)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': unet1d_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, filename=best_model_tar)

        if (epoch - epoch_loaded) in iters_to_copy:
            target_folder = f'/BCM_Mamba_transfer_4k_{epoch} (Please enter the path that you want to save the model)'
            os.makedirs(target_folder, exist_ok=True)
            shutil.copy(best_model_pth, os.path.join(target_folder, 'best_model.pth'))
            shutil.copy(best_model_tar, os.path.join(target_folder, 'best_model.pth.tar'))
            print(f"Copied best model files to {target_folder} at iteration {epoch}")

        print_lr(optimizer)

if __name__ == "__main__":
    main()