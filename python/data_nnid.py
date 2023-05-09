"""
1. Synthesize audio data
2. Feature extraction for audio data.
"""
import os
import time
import argparse
import re
import multiprocessing
import logging
import random
import numpy as np
import boto3
import soundfile as sf
import sounddevice as sd
import librosa
import yaml
from yaml.loader import SafeLoader
from nnsp_pack import tfrecord_converter_nnid
from nnsp_pack.feature_module import FeatureClass, display_stft
from nnsp_pack import add_noise
from nnsp_pack import boto3_op

DEBUG = True
UPLOAD_TFRECORD_S3 = False
DOWLOAD_DATA = False
MAX_FRAMES = 350
DELAY_FRAMES = 10
NUM_SENTS = 7
NUM_GROUP_PPLS = 64
NOISE_TYPES = [
        'ESC-50-MASTER',
        'wham_noise',
        "social_noise",
        'FSD50K',
        'musan',
        'traffic' ]
if DEBUG:
    SNR_DBS_MIN_MAX = [100]
else:
    SNR_DBS_MIN_MAX = [5, 10, 15, 20, 40]
if UPLOAD_TFRECORD_S3:
    print('uploading tfrecords to s3 will slow down the process')
S3_BUCKET = "ambiqai-speech-commands-dataset"
S3_PREFIX = "tfrecords"

params_audio = {
    'win_size'      : 240,
    'hop'           : 80,
    'len_fft'       : 256,
    'sample_rate'   : 8000,
    'nfilters_mel'  : 22}

def download_data():
    """
    download data
    """
    audio_lists = [
        'data/test_files_vad.csv',
        'data/train_files_vad.csv',
        'data/noise_list.csv']
    s3 = boto3.client('s3')
    boto3_op.s3_download(S3_BUCKET, audio_lists)
    return s3

class FeatMultiProcsClass(multiprocessing.Process):
    """
    FeatMultiProcsClass use multiprocesses
    to run several processes of feature extraction in parallel
    """
    def __init__(self, id_process,
                 name, src_list, train_set, ntypes,
                 snr_dbs_min_max, success_dict,
                 params_audio_def,
                 num_processes = 8):

        multiprocessing.Process.__init__(self)
        self.success_dict = success_dict
        self.id_process         = id_process
        self.name               = name
        self.src_list           = src_list
        self.params_audio_def   = params_audio_def
        self.num_processes      = num_processes
        self.feat_inst      = FeatureClass(
                                win_size        = params_audio_def['win_size'],
                                hop             = params_audio_def['hop'],
                                len_fft         = params_audio_def['len_fft'],
                                sample_rate     = params_audio_def['sample_rate'],
                                nfilters_mel    = params_audio_def['nfilters_mel'])

        self.train_set          = train_set
        self.ntypes              = ntypes
        self.snr_dbs_min_max    = snr_dbs_min_max
        self.names=[]
        if DEBUG:
            self.cnt = 0

    def run(self):
        #      threadLock.acquire()
        print("Running " + self.name)

        self.convert_tfrecord(
                    self.src_list,
                    self.id_process)

    def convert_tfrecord(
            self,
            ppls,
            id_process):
        """
        convert np array to tfrecord
        """
        outlist = [None] * len(ppls)
        for p, fnames in enumerate(ppls):
            outlist[p] = [None] * NUM_SENTS
            random.shuffle(fnames)
            for i, fname in enumerate(fnames[:NUM_SENTS]):
                outlist[p][i] = []
                for ntype in self.ntypes:
                    ntype0 = re.sub(r'/', '_', ntype)
                    noise_files_train = f'data/noise_list/train_noiselist_{ntype0}.csv'
                    noise_files_test = f'data/noise_list/test_noiselist_{ntype0}.csv'
                    with open(noise_files_train) as file: # pylint: disable=unspecified-encoding
                        lines = file.readlines()
                    lines_tr = [line.strip() for line in lines]

                    with open(noise_files_test) as file: # pylint: disable=unspecified-encoding
                        lines = file.readlines()
                    lines_te = [line.strip() for line in lines]
                    noise_files = { 'train' : lines_tr,
                                    'test'  : lines_te}
                    for snr_db in self.snr_dbs_min_max:
                        if self.id_process == self.num_processes - 1:
                            print(f"\r{p:>5}/{len(ppls)} {ntype:<15}, snr_db = {snr_db:>3}", end="")

                        success = 1
                        stimes = []
                        etimes = []
                        targets = []
                        speech = np.empty(0)
                        pattern = r'(\.wav$|\.flac$)'

                        bks = fname.strip().split(',')
                        wavpath = bks[0]
                        stime = int(bks[1])         # start time
                        etime = int(bks[2])         # end time
                        tfrecord = re.sub(
                            pattern,
                            '.tfrecord',
                            re.sub(r'wavs', S3_PREFIX, wavpath))
                        try:
                            audio, sample_rate = sf.read(wavpath)
                        except :# pylint: disable=bare-except
                            success = 0
                            print(f"Reading the {wavpath} fails ")
                            break
                        else:
                            if audio.ndim > 1:
                                audio=audio[:,0]
                            if sample_rate > self.feat_inst.sample_rate:
                                audio = librosa.resample(
                                        audio,
                                        orig_sr=sample_rate,
                                        target_sr=self.feat_inst.sample_rate)

                            elif sample_rate < self.feat_inst.sample_rate:
                                print(f"{wavpath}: sampling rate < target fs={self.feat_inst.sample_rate}") # pylint: disable=line-too-long

                            # decorate speech
                            speech = np.zeros((MAX_FRAMES*self.feat_inst.hop,),dtype=np.float32)
                            speech_orig = audio[stime : etime]
                            etime = (150 - DELAY_FRAMES)*80
                            zeros = np.zeros((DELAY_FRAMES*80,), dtype=np.float32)
                            speech_orig = np.concatenate((speech_orig, zeros))
                            if len(speech_orig) > 150 * 80:
                                speech_orig = speech_orig[-150*80:]
                                speech = speech_orig
                                stime=0
                            elif len(speech_orig) < 150 * 80:
                                stime = 150*80-len(speech_orig)
                                zeros = np.zeros((stime,), dtype=np.float32)
                                speech = np.concatenate((zeros, speech_orig))
                            else:
                                stime = 0
                                speech = speech_orig
                            target = 1
                            stimes += [stime]
                            etimes += [etime]
                            targets += [target]

                        if success:
                            stimes  = np.array(stimes)
                            etimes  = np.array(etimes)
                            targets = np.array(targets)
                            start_frames    = (stimes / self.params_audio_def['hop']) + 2
                            start_frames    = start_frames.astype(np.int32)
                            end_frames      = (etimes / self.params_audio_def['hop']) + 2
                            end_frames      = end_frames.astype(np.int32)
                            # add noise to sig
                            noise = add_noise.get_noise(
                                        noise_files[self.train_set],
                                        len(speech),
                                        self.feat_inst.sample_rate)
                            audio = add_noise.add_noise(
                                        speech,
                                        noise,
                                        snr_db,
                                        stime,
                                        etime,
                                        return_all = False,
                                        amp_min=0.01,
                                        amp_max=0.95)
                            # feature extraction of sig
                            spec, _, feat, _ = self.feat_inst.block_proc(audio)
                            ntype = re.sub('/','_', ntype)
                            tfrecord = re.sub(  r'\.tfrecord$',
                                                f'_snr{snr_db}dB_{ntype}.tfrecord',
                                                tfrecord)
                            os.makedirs(os.path.dirname(tfrecord), exist_ok=True)
                            try:
                                timesteps, _  = feat.shape
                                width_targets = end_frames - start_frames + 1
                                tfrecord_converter_nnid.make_tfrecord( # pylint: disable=too-many-function-args
                                                    tfrecord,
                                                    feat)

                            except: # pylint: disable=bare-except
                                print(f"Thread-{id_process}: {i}, processing {tfrecord} failed")
                            else:
                                # self.success_dict[self.id_process] += [tfrecord]
                                outlist[p][i] += [tfrecord]
                                # since tfrecord file starts: data/tfrecords/speakers/...
                                # strip the leading "data/" when uploading
                                if UPLOAD_TFRECORD_S3:
                                    s3.upload_file(tfrecord, S3_BUCKET, tfrecord)
                                else:
                                    pass
                            if DEBUG:
                                os.makedirs('test_wavs', exist_ok=True)
                                sd.play(audio, self.feat_inst.sample_rate)
                                print(fnames[i])
                                print(targets)
                                print(audio)
                                flabel = np.zeros(spec.shape[0])
                                tmp = zip(start_frames, end_frames, targets)
                                for start_frame, end_frame, target in tmp:
                                    flabel[start_frame: end_frame] = target
                                display_stft(
                                    audio, spec.T, feat.T,
                                    self.feat_inst.sample_rate,
                                    label_frame=flabel)

                                sf.write(f'test_wavs/speech_{self.cnt}.wav',
                                        audio,
                                        self.feat_inst.sample_rate)
                                sf.write(f'test_wavs/speech_{self.cnt}_ref.wav',
                                        speech,
                                        self.feat_inst.sample_rate)

                                self.cnt = self.cnt + 1

        self.success_dict[self.id_process] = outlist

def main(args):
    """
    main function to generate all training and testing data
    """
    if DOWLOAD_DATA:
        s3 = download_data()
    if args.wandb_track:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="data-update")
        wandb.config.update(args)

    train_sets = ["train", "test"]
    # Prepare noise dataset, train and test sets
    os.makedirs('data/noise_list', exist_ok=True)
    for ntype in NOISE_TYPES:
        if ntype=='wham_noise':
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'
                if set0 == 'train':
                    lst_ns = add_noise.get_noise_files_new('wham_noise/tr')
                else:
                    lst_ns = add_noise.get_noise_files_new('wham_noise/cv')
                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')
        elif ntype in {'FSD50K','ESC-50-MASTER'}:
            with open(f'wavs/noise/{ntype}/non_speech.csv', 'r') as file: # pylint: disable=unspecified-encoding
                lines = file.readlines()
            random.shuffle(lines)
            lst_ns={}
            start = int(len(lines) / 5)

            lst_ns['train'] = lines[start:]
            lst_ns['test'] = lines[:start]

            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}')
                    if ntype=='ESC-50-MASTER':
                        lst_must = add_noise.get_noise_files_new('others/must')
                        for name in lst_must:
                            name = re.sub(r'\\', '/', name)
                            file.write(f'{name}\n')
        elif ntype in {'musan'}:
            lst_music = add_noise.get_noise_files_new('musan/music')
            lst_noise = add_noise.get_noise_files_new('musan/noise')
            lines = lst_music + lst_noise
            random.shuffle(lines)
            lst_ns={}
            start = int(len(lines) / 5)
            lst_ns['test'] = lines[:start]
            lst_ns['train'] = lines[start:]
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')

        elif ntype in {'social_noise', 'traffic'}:
            lst_ns={}
            lst_ns['test'] = add_noise.get_noise_files_new(f'{ntype}/test')
            lst_ns['train'] = add_noise.get_noise_files_new(f'{ntype}/train')
            for set0 in ['train', 'test']:
                noise_files_lst = f'data/noise_list/{set0}_noiselist_{ntype}.csv'

                with open(noise_files_lst, 'w') as file: # pylint: disable=unspecified-encoding
                    for name in lst_ns[set0]:
                        name = re.sub(r'\\', '/', name)
                        file.write(f'{name}\n')
        else:
            ntype0 = re.sub(r'/', '_', ntype)
            noise_files_train = f'data/noise_list/train_noiselist_{ntype0}.csv'
            noise_files_test = f'data/noise_list/test_noiselist_{ntype0}.csv'
            lst_ns = add_noise.get_noise_files_new(ntype)
            random.shuffle(lst_ns)
            start = int(len(lst_ns) / 5)
            with open(noise_files_train, 'w') as file: # pylint: disable=unspecified-encoding
                for name in lst_ns[start:]:
                    name = re.sub(r'\\', '/', name)
                    file.write(f'{name}\n')

            with open(noise_files_test, 'w') as file:  # pylint: disable=unspecified-encoding
                for name in lst_ns[:start]:
                    name = re.sub(r'\\', '/', name)
                    file.write(f'{name}\n')
    target_files = { 'train': args.train_dataset_path,
                     'test' : args.test_dataset_path}
    tot_success_dict = {'train': [], 'test': []}

    for train_set in train_sets:
        with open(target_files[train_set]) as file: # pylint: disable=unspecified-encoding
            filepaths_all = yaml.load(file, Loader=SafeLoader)
            len0 = NUM_GROUP_PPLS * int(len(filepaths_all) / NUM_GROUP_PPLS)
            filepaths = filepaths_all[:len0]
        blk_size = int(np.floor(len(filepaths) / args.num_procs))
        sub_src = []
        for i in range(args.num_procs):
            idx0 = i * blk_size
            if i == args.num_procs - 1:
                sub_src += [filepaths[idx0:]]
            else:
                sub_src += [filepaths[idx0:blk_size+idx0]]
        manager = multiprocessing.Manager()
        success_dict = manager.dict({i: [] for i in range(args.num_procs)})
        print(f'{train_set} set running:, snr = {SNR_DBS_MIN_MAX} db')

        processes = [
            FeatMultiProcsClass(
                    i, f"Thread-{i}",
                    sub_src[i],
                    train_set,
                    NOISE_TYPES,
                    SNR_DBS_MIN_MAX,
                    success_dict,
                    params_audio_def = params_audio,
                    num_processes = args.num_procs)
                        for i in range(args.num_procs)]

        start_time = time.time()

        if DEBUG:
            for proc in processes:
                proc.run()
        else:
            for proc in processes:
                proc.start()

            for proc in processes:
                proc.join()
            print(f"\nTime elapse {time.time() - start_time} sec")

        if args.wandb_track:
            data = wandb.Artifact(
                S3_BUCKET + "-tfrecords",
                type="dataset",
                description="tfrecords of speech command dataset")
            data.add_reference(f"s3://{S3_BUCKET}/{S3_PREFIX}", max_objects=31000)
            run.log_artifact(data)

        for lst in success_dict.values():
            tot_success_dict[train_set] += lst

    if not DEBUG:
        for train_set in train_sets:
            with open(f'data/{train_set}_tfrecords_nnid.yaml', 'w') as file: # pylint: disable=unspecified-encoding
                yaml.dump(tot_success_dict[train_set], file)
                # for tfrecord in tot_success_dict[train_set]:
                #     tfrecord = re.sub(r'\\', '/', tfrecord)
                #     file.write(f'{tfrecord}\n')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    argparser = argparse.ArgumentParser(
        description='Generate TFrecord formatted input data from a raw speech commands dataset')

    argparser.add_argument(
        '-tr',
        '--train_dataset_path',
        default = 'data/nnid_train.yaml',
        help    = 'path to train data file')

    argparser.add_argument(
        '-tt',
        '--test_dataset_path',
        default = 'data/nnid_test.yaml',
        help    = 'path to test data file')

    argparser.add_argument(
        '-ss',
        '--size_train',
        type    = int,
        default = 10000,
        help='')

    argparser.add_argument(
        '-n',
        '--num_procs',
        type    = int,
        default = 8,
        help='How many processor cores to use for execution')

    argparser.add_argument(
        '-w',
        '--wandb_track',
        default = False,
        help    = 'Enable tracking of this run in Weights&Biases')

    argparser.add_argument(
        '--wandb_project',
        type    = str,
        default = 'vad',
        help='Weights&Biases project name')

    argparser.add_argument(
        '--wandb_entity',
        type    = str,
        default = 'ambiq',
        help    = 'Weights&Biases entity name')

    args_ = argparser.parse_args()
    main(args_)
