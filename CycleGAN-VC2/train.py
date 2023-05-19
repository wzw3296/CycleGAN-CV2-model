
import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle
import soundfile as sf

import preprocess
from trainingDataset import trainingDataset
from model_tf import Generator, Discriminator
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

class CycleGANTraining(object):
    def __init__(self,
                 logf0s_normalization,
                 mcep_normalization,
                 src_coded_norm,
                 target_coded_norm,
                 model_checkpoint,
                 validation_src,
                 output_src,
                 validation_target,
                 output_target,
                 restart_training_at=None):
        self.start_epoch = 0
        self.num_epochs = 250   #200000
        self.mini_batch_size = 1  # 1
        self.dataset_A = self.loadPickleFile(src_coded_norm)
        self.dataset_B = self.loadPickleFile(target_coded_norm)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        logf0s_normalization = np.load(logf0s_normalization)
        self.log_f0s_mean_A = logf0s_normalization['mean_A']
        self.log_f0s_std_A = logf0s_normalization['std_A']
        self.log_f0s_mean_B = logf0s_normalization['mean_B']
        self.log_f0s_std_B = logf0s_normalization['std_B']

        mcep_normalization = np.load(mcep_normalization)
        self.coded_sps_A_mean = mcep_normalization['mean_A']
        self.coded_sps_A_std = mcep_normalization['std_A']
        self.coded_sps_B_mean = mcep_normalization['mean_B']
        self.coded_sps_B_std = mcep_normalization['std_B']

        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)

        criterion_mse = torch.nn.MSELoss()


        g_params = list(self.generator_A2B.parameters()) + \
                   list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + \
                   list(self.discriminator_B.parameters())

        self.generator_lr = 2e-4  # 0.0002
        self.discriminator_lr = 1e-4  # 0.0001

        self.generator_lr_decay = self.generator_lr / 2000 #200000
        self.discriminator_lr_decay = self.discriminator_lr / 2000 #200000

        self.start_decay = 10000  # 200000

        self.generator_optimizer = torch.optim.Adam(
            g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(
            d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))

        self.modelCheckpoint = model_checkpoint
        os.makedirs(self.modelCheckpoint, exist_ok=True)

        self.validation_src = validation_src
        self.output_src = output_src
        os.makedirs(self.output_src, exist_ok=True)
        self.validation_target = validation_target
        self.output_target = output_target
        os.makedirs(self.output_target, exist_ok=True)

        self.generator_loss_store = []
        self.discriminator_loss_store = []

        self.file_name = 'log_store_non_sigmoid.txt'

        if restart_training_at is not None:
            self.start_epoch = self.loadModel(restart_training_at)
            print("Training resumed")

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()

            cycle_loss_lambda = 10
            identity_loss_lambda = 5

            n_samples = len(self.dataset_A)

            dataset = trainingDataset(src_data=self.dataset_A,
                                      target_data=self.dataset_B,
                                      n_frames=128)
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

            pbar = tqdm(enumerate(train_loader))
            for i, (real_A, real_B) in enumerate(train_loader):
                num_iterations = (n_samples // self.mini_batch_size) * epoch + i

                if num_iterations > 10000:
                    identity_loss_lambda = 0
                if num_iterations > self.start_decay:
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(
                        self.generator_optimizer, name='discriminator')

                real_A = real_A.to(self.device).float()
                real_B = real_B.to(self.device).float()


                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)


                d_fake_cycle_A = self.discriminator_A(cycle_A)
                d_fake_cycle_B = self.discriminator_B(cycle_B)

                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                identiyLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                                 cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
                self.generator_loss_store.append(generator_loss.item())


                self.reset_grad()
                generator_loss.backward()

                self.generator_optimizer.step()

                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                cycled_B = self.generator_A2B(generated_A)
                d_cycled_B = self.discriminator_B(cycled_B)

                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)

                cycled_A = self.generator_B2A(generated_B)
                d_cycled_A = self.discriminator_A(cycled_A)

                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                d_loss_A_cycled = torch.mean((0 - d_cycled_A) ** 2)
                d_loss_B_cycled = torch.mean((0 - d_cycled_B) ** 2)
                d_loss_A_2nd = (d_loss_A_real + d_loss_A_cycled) / 2.0
                d_loss_B_2nd = (d_loss_B_real + d_loss_B_cycled) / 2.0

                d_loss = (d_loss_A + d_loss_B) / 2.0 + (d_loss_A_2nd + d_loss_B_2nd) / 2.0
                self.discriminator_loss_store.append(d_loss.item())

                self.reset_grad()
                d_loss.backward()

                self.discriminator_optimizer.step()

                if (i + 1) % 2 == 0:
                    pbar.set_description(
                        "Iter:{} Generator Loss:{:.4f} Discrimator Loss:{:.4f} GA2B:{:.4f} GB2A:{:.4f} G_id:{:.4f} G_cyc:{:.4f} D_A:{:.4f} D_B:{:.4f}".format(
                            num_iterations,
                            generator_loss.item(),
                            d_loss.item(),
                            generator_loss_A2B,
                            generator_loss_B2A,
                            identiyLoss,
                            cycleLoss,
                            d_loss_A,
                            d_loss_B))


            if epoch % 10 == 0 and epoch != 0:
                end_time = time.time()
                store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                    epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
                self.store_to_file(store_to_file)
                print("Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                    epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch))

                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(store_to_file)
                self.saveModelCheckPoint(epoch, '{}'.format(
                    self.modelCheckpoint + '_CycleGAN_CheckPoint'))
                print("Model Saved!")

            if epoch % 10 == 0 and epoch != 0:
                validation_start_time = time.time()
                self.validation_for_A_dir()
                validation_end_time = time.time()
                store_to_file = "Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time)
                self.store_to_file(store_to_file)
                print("Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time))

    def validation_for_A_dir(self):
        num_mcep = 36
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_src = self.validation_src
        output_src = self.output_src

        print("Generating Validation Data B from A...")
        for file in os.listdir(validation_src):
            filePath = os.path.join(validation_src, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = preprocess.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = preprocess.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = preprocess.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = preprocess.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                                 self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = preprocess.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = preprocess.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            # librosa.write(os.path.join(output_src, os.path.basename(file)),
            #               wav_transformed,
            #               sampling_rate)
            # librosa.output.write_wav(path=os.path.join(output_src, os.path.basename(file)),
            #                          y=wav_transformed,
            #                          sr=sampling_rate)
            sf.write(os.path.join(output_src, os.path.basename(file)),
                     wav_transformed,
                     sampling_rate)

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f) #torch.save(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch


if __name__ == '__main__':
    logf0s_normalization = './cache/logf0s_normalization.npz'
    mcep_normalization = './cache/mcep_normalization.npz'
    src_coded_norm = './cache/src_coded_norm.pickle'
    target_coded_norm = './cache/target_coded_norm.pickle'
    model_checkpoint = './model_checkpoint/'
    #resume_training_at = './model_checkpoint/_CycleGAN_CheckPoint'
    resume_training_at = None
    validation_src = './data/sam/'
    output_src = './converted_data/sam'
    validation_target = './data/bea/'
    output_target = './converted_data/bea'

    cycleGAN = CycleGANTraining(logf0s_normalization=logf0s_normalization,
                                mcep_normalization=mcep_normalization,
                                src_coded_norm=src_coded_norm,
                                target_coded_norm=target_coded_norm,
                                model_checkpoint=model_checkpoint,
                                validation_src=validation_src,
                                output_src=output_src,
                                validation_target=validation_target,
                                output_target=output_target,
                                restart_training_at=resume_training_at)
    cycleGAN.train()
