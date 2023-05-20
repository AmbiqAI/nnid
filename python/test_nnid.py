"""
Test trained NN model using wavefile as input
"""
import os
import re
import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
import librosa
from data_nnid_ti import params_audio as PARAM_AUDIO, MAX_FRAMES
from nnsp_pack.nn_infer_nnid import NNIDClass
from nnsp_pack.pyaudio_animation import AudioShowClass
from nnsp_pack.nn_infer_vad import VadClass

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False
EPS = 10**-5
def cos_score(vec0, vec1):
    """
    calculate cosine score
    """
    norm_vec0 = np.sum(vec0**2)
    norm_vec1 = np.sum(vec1**2)
    score = np.sum(vec0 * vec1) / np.sqrt(np.maximum(norm_vec0 * norm_vec1, EPS))
    return score

def main(args):
    """main function"""
    epoch_loaded    = int(args.epoch_loaded)
    quantized       = args.quantized
    speaker         = args.speaker
    threshold       = args.threshold
    num_eroll       = args.num_eroll

    # load vad class
    nn_arch_vad = 'nn_arch/def_vad_nn_arch24_moredata.txt'
    epoch_loaded_vad = 253
    params_audio_vad = {
        'win_size'      : 240,
        'hop'           : 80,
        'len_fft'       : 256,
        'sample_rate'   : 8000,
        'nfilters_mel'  : 22}

    vad_inst = VadClass(
                    nn_arch_vad,
                    epoch_loaded_vad,
                    params_audio_vad,
                    quantized       = True,
                    show_histogram  = False,
                    np_inference    = False
                    )

    # recording ref embd vector
    os.makedirs(f"./test_wavs/{speaker}", exist_ok=True)
    fname_embd = f"test_wavs/{speaker}/embedding.npy"
    for i in range(num_eroll):
        wavefile = f"./test_wavs/{speaker}/record_{i}.wav"
        audio_handle = AudioShowClass(
            record_seconds=6,
            wave_output_filename=wavefile,
            non_stop=False,
            id_enroll=i)
        if audio_handle.is_new_record():
            if os.path.exists(fname_embd):
                os.remove(fname_embd)
    embds = []
    nnid_inst = NNIDClass(
            args.nn_arch,
            epoch_loaded,
            PARAM_AUDIO,
            quantized,
            show_histogram  = SHOW_HISTOGRAM,
            np_inference    = NP_INFERENCE)

    # generate ref embd vector
    if not os.path.exists(fname_embd):
        for i in range(num_eroll):
            wavefile = f"test_wavs/{speaker}/record_{i}.wav"
            data, sample_rate = sf.read(wavefile)
            if data.ndim > 1:
                data = data[:,0]
            # vad testing
            if sample_rate > params_audio_vad['sample_rate']:
                data_vad = librosa.resample(
                    data,
                    orig_sr=sample_rate,
                    target_sr=params_audio_vad['sample_rate'])
            vad_inst.reset()
            _, start = vad_inst.blk_proc(data_vad, thresh_prob =0.5)

            # generate embedding
            nnid_inst.reset()
            if sample_rate > PARAM_AUDIO['sample_rate']:
                data = librosa.resample(
                    data,
                    orig_sr=sample_rate,
                    target_sr=PARAM_AUDIO['sample_rate'])
            data = data[start * PARAM_AUDIO['hop']: (start + MAX_FRAMES) * PARAM_AUDIO['hop']]
            sd.play(data, PARAM_AUDIO['sample_rate'])

            os.makedirs("test_results", exist_ok=True)
            name_wavout = 'test_results/output_' + os.path.basename(wavefile)
            embd = nnid_inst.blk_proc(
                data,
                name_wavout=name_wavout)
            embds += [embd]
        embds = np.array(embds)
        embds /= np.sqrt(np.maximum(np.sum(embds**2, keepdims=True, axis=-1),EPS))
        embd_spk = np.mean(embds, axis=0)
        np.save(fname_embd, embd_spk)
    else:
        embd_spk = np.load(fname_embd)

    # recording testing embd vector
    wavefile_spk = f'test_wavs/{speaker}/speech_test.wav'
    AudioShowClass(
        record_seconds=6,
        wave_output_filename=wavefile_spk,
        non_stop=False)

    # generate testing embd vector
    wavefiles = [wavefile_spk]
    for root, _, files in os.walk("test_wavs/test_set"):
        for file in files:
            if re.search(r"(\.wav$|\.flac$)", file):
                wavfile = os.path.join(root, file)
                wavfile = re.sub(r"\\", '/', wavfile)
                wavefiles += [wavfile]

    for i, wavefile in enumerate(wavefiles):
        print("/-------------------------------------------/")
        print(f"wavefile : {wavefile}")

        # vad testing
        data, sample_rate = sf.read(wavefile)
        if data.ndim > 1:
            data = data[:,0]
        if sample_rate > params_audio_vad['sample_rate']:
            data_vad = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=params_audio_vad['sample_rate'])
        vad_inst.reset()
        _, start = vad_inst.blk_proc(
            data_vad, thresh_prob =0.5, show_fig=False)
        # speaker verification
        nnid_inst.reset()
        if sample_rate > PARAM_AUDIO['sample_rate']:
            data = librosa.resample(
                data,
                orig_sr=sample_rate,
                target_sr=PARAM_AUDIO['sample_rate'])

        data = data[start * PARAM_AUDIO['hop'] : (start+MAX_FRAMES) * PARAM_AUDIO['hop']]
        if i == 0:
            wavefile_chop = re.sub(r"\.wav", "_chop.wav", wavefile)
            sf.write(f"{wavefile_chop}", data, PARAM_AUDIO['sample_rate'])
        sd.play(data, PARAM_AUDIO['sample_rate'])

        os.makedirs("test_results", exist_ok=True)
        name_wavout = 'test_results/output_' + os.path.basename(wavefile)
        embd_test = nnid_inst.blk_proc(
            data,
            name_wavout=name_wavout,
            show_stft=False)

        embd_spk = (embd_spk * 2**15).astype(np.int32)
        embd_test = (embd_test * 2**15).astype(np.int32)

        score = cos_score(embd_spk / 2**15, embd_test / 2**15)
        print(f"score = {score}")

        if score >= threshold:
            print(f"Yes, {speaker} is verified")
        else:
            print(f"No, {speaker} is not verified")

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
        default = "Tom",
        type=str,
        help    = "speaker to enroll")

    argparser.add_argument(
        '-t',
        '--threshold',
        default = 0.80,
        type=float,
        help    = "threshold for spk verification")

    argparser.add_argument(
        '-ne',
        '--num_eroll',
        default = 3,
        type=int,
        help    = "number of sentences to enroll NNID")

    argparser.add_argument(
        '-q',
        '--quantized',
        default = True,
        type=bool,
        help='is post quantization?')

    argparser.add_argument(
        '--epoch_loaded',
        default= 102,
        help='starting epoch')

    main(argparser.parse_args())
