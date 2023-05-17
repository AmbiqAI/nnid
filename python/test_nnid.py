"""
Test trained NN model using wavefile as input
"""
import os
import re
import time
import argparse
import wave
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
from nnsp_pack.nn_activation import softmax # pylint: disable=no-name-in-module
from nnsp_pack.feature_module import display_stft
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer import NNInferClass
from data_nnid_ti import params_audio as PARAM_AUDIO, MAX_FRAMES

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class NNIDClass(NNInferClass):
    """
    Class to handle VAD model
    """
    def __init__(
            self,
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized=False,
            show_histogram=False,
            np_inference=False):

        super().__init__(
            nn_arch,
            epoch_loaded,
            params_audio,
            quantized,
            show_histogram,
            np_inference)

        self.cnt_vad_trigger = np.zeros(2, dtype=np.int32)
        self.vad_prob = 0.0
        self.vad_trigger = 0

    def reset(self):
        """
        Reset s2i instance
        """
        super().reset()
        self.vad_trigger = 0
        self.vad_prob = 0.0
        self.cnt_vad_trigger *= 0

    def post_nn_infer(self, nn_output, thresh_prob=0.3):
        """
        post nn inference
        """
        self.vad_prob = softmax(nn_output)[1]
        if self.vad_prob > thresh_prob:
            self.vad_trigger = 1
        else:
            self.vad_trigger = 0
        if self.vad_trigger == 0:
            self.cnt_vad_trigger *= 0
        else:
            if self.cnt_vad_trigger[self.vad_trigger] == 0:
                self.cnt_vad_trigger *= 0
            self.cnt_vad_trigger[self.vad_trigger] += 1

    def frame_proc(self, data_frame):
        """
        VAD frame process
        Output:
                Trigger
        """
        feat, spec = self.frame_proc_np(data_frame)
        return self.vad_trigger

    def blk_proc(
            self,
            data,
            name_wavout='test_results/output.wav',
            show_stft=False):
        """
        NN process for several frames
        """
        params_audio = self.params_audio
        file = wave.open(name_wavout, "wb")
        file.setnchannels(2)
        file.setsampwidth(2)
        file.setframerate(params_audio['sample_rate'])

        bks = int(len(data) / params_audio['hop'])
        feats = []
        specs = []
        embds = []
        stime = time.time()
        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]

            feat, spec, embd = self.frame_proc_tf(data_frame, return_all=True)

            if self.cnt_vad_trigger[self.vad_trigger] >= 1:
                print(f'\rFrame {i}: trigger', end="")
            else:
                print(f'\rFrame {i}:', end='')

            feats += [feat]
            specs += [spec]
            embds += [embd]
            self.count_run = (self.count_run + 1) % self.num_dnsampl
        etime = time.time()
        print(f" ave {(etime-stime)/bks * 1000:2f} ms/inf")
        feats = np.array(feats)
        specs = np.array(specs)
        embds = np.array(embds)

        out = np.empty(
                (data.size,),
                dtype=data.dtype)
        out = data
        out = np.floor(out * 2**15).astype(np.int16)
        file.writeframes(out.tobytes())
        file.close()

        if show_stft:
            display_stft(
                data,
                specs.T,
                feats.T,
                embds.T,
                sample_rate=PARAM_AUDIO['sample_rate'])
        return embds[-1]

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    speaker         = args.speaker
    threshold       = args.threshold

    os.makedirs(f"./test_wavs/{speaker}", exist_ok=True)

    for i in range(3):
        wavefile = f"./test_wavs/{speaker}/record_{i}.wav"
        AudioShowClass(
            record_seconds=6,
            wave_output_filename=wavefile,
            non_stop=False,
            id_enroll=i)

    embds = []
    nnid_inst = NNIDClass(
            args.nn_arch,
            epoch_loaded,
            PARAM_AUDIO,
            quantized,
            show_histogram  = SHOW_HISTOGRAM,
            np_inference    = NP_INFERENCE)
    fname_embd = f"test_wavs/{speaker}/embedding.npy"
    if not os.path.exists(fname_embd):
        for i in range(3):
            wavefile = f"test_wavs/{speaker}/record_{i}.wav"
            data, sample_rate = sf.read(wavefile)
            nnid_inst.reset()
            if data.ndim > 1:
                data = data[:,0]
            if sample_rate > PARAM_AUDIO['sample_rate']:
                data = librosa.resample(
                    data,
                    orig_sr=sample_rate,
                    target_sr=PARAM_AUDIO['sample_rate'])

            data = data[-MAX_FRAMES * PARAM_AUDIO['hop']:]

            sd.play(data, PARAM_AUDIO['sample_rate'])

            os.makedirs("test_results", exist_ok=True)
            name_wavout = 'test_results/output_' + os.path.basename(wavefile)
            embd = nnid_inst.blk_proc(data, name_wavout=name_wavout)
            embds += [embd]
        embds = np.array(embds)
        embds /= np.sqrt(np.maximum(np.sum(embds**2, keepdims=True, axis=-1),10**-5))
        embd_spk = np.mean(embds, axis=0)
        np.save(fname_embd, embd_spk)
    else:
        embd_spk = np.load(fname_embd)

    wavefile_spk = f'test_wavs/{speaker}/speech_test.wav'
    AudioShowClass(
        record_seconds=6,
        wave_output_filename=wavefile_spk,
        non_stop=False)

    wavefiles = [wavefile_spk]
    for root, _, files in os.walk("test_wavs/test_set"):
        for file in files:
            if re.search(r"(\.wav$|\.flac$)", file):
                wavfile = os.path.join(root, file)
                wavfile = re.sub(r"\\", '/', wavfile)
                wavefiles += [wavfile]

    for wavefile in wavefiles:
        print("/-------------------------------------------/")
        print(f"wavefile : {wavefile}")
        data, sample_rate = sf.read(wavefile)
        nnid_inst.reset()
        if data.ndim > 1:
            data = data[:,0]
        if sample_rate > PARAM_AUDIO['sample_rate']:
            data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=PARAM_AUDIO['sample_rate'])

        data = data[-MAX_FRAMES * PARAM_AUDIO['hop']:]

        sd.play(data, PARAM_AUDIO['sample_rate'])

        os.makedirs("test_results", exist_ok=True)
        name_wavout = 'test_results/output_' + os.path.basename(wavefile)
        embd_test = nnid_inst.blk_proc(
            data,
            name_wavout=name_wavout,
            show_stft=False)
        score = cos_score(embd_spk, embd_test)
        print(f"score = {score}")

        if score >= threshold:
            print(f"Yes, {speaker} is verified")
        else:
            print(f"No, {speaker} is not verified")

def cos_score(vec0, vec1):
    """
    calculate cosine score
    """
    norm_vec0 = np.sum(vec0**2)
    norm_vec1 = np.sum(vec1**2)
    score = np.sum(vec0 * vec1) / np.sqrt(np.maximum(norm_vec0 * norm_vec1, 10**-5))
    return score

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description='Testing trained Speaker ID model')

    argparser.add_argument(
        '-a',
        '--nn_arch',
        default='nn_arch/def_id_nn_arch100_ti.txt',
        help='nn architecture')

    argparser.add_argument(
        '-s',
        '--speaker',
        default = "paul",
        type=str,
        help    = "speaker to enroll")

    argparser.add_argument(
        '-t',
        '--threshold',
        default = 0.70,
        type=float,
        help    = "threshold for spk verification")

    argparser.add_argument(
        '-q',
        '--quantized',
        default = False,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 80,
        help='starting epoch')

    main(argparser.parse_args())
