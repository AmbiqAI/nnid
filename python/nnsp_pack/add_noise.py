"""
This module deals with sythesizing speech and noise data
"""
import os
import re
import logging
import soundfile as sf
import numpy as np
import librosa

def get_power(data):
    """Calculate power of data"""
    return np.mean(data**2)

def add_noise(data0, noise, snr_db, stime, etime,
              return_all=False,
              amp_min = 0.01,
              amp_max = 0.95):
    """Synthesize noise and speech"""
    pw_data = get_power(data0[stime:etime])
    pw_noise = get_power(noise)
    snr = 10**(snr_db/10)
    if pw_data != 0:
        data = data0 / np.sqrt(pw_data)
    else:
        data = data0
    if pw_noise != 0 and snr != 0:
        noise = noise / np.sqrt(pw_noise) / np.sqrt(snr)
    output = data + noise
    max_val = np.abs(output).max()
    prob = np.random.uniform(amp_min, amp_max, 1)

    gain    = prob / (max_val + 10**-5)
    output  = output * gain
    data    = data   * gain
    noise   = noise  * gain
    # if snr_dB_improved:
    #     gain0 = 10**(snr_dB_improved / 20)
    #     data = data + noise / gain0
    # print(output)
    if return_all:
        return output, data
    else:
        return output

def get_noise_files_new(path_noise_folder):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{path_noise_folder}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise_files(files_list, noise_type):
    """Fetch all of noise files"""
    lst = []
    for root, _, files in os.walk(f'wavs/noise/{noise_type}/{files_list}'):
        for file in files:
            if re.search(r'wav$', file):
                lst += [os.path.join(root, file.strip())]
    return lst

def get_noise(
        fnames,
        length = 16000 * 5,
        target_sampling_rate=16000):
    """Random pick ONE of noise from fnames"""
    len0 = len(fnames)
    rand_idx = np.random.randint(0, len0)

    try:
        noise, sample_rate_in = sf.read(fnames[rand_idx])
    except: # pylint: disable=W0702
        logging.debug('reading noise file %s failed', fnames[rand_idx] )
        noise = np.random.randn(target_sampling_rate).astype(np.float32) * 0.1
    else:
        if noise.ndim > 1:
            noise = noise[:,0]

        if sample_rate_in > target_sampling_rate:
            try:
                noise = librosa.resample(
                            noise,
                            orig_sr=sample_rate_in,
                            target_sr=target_sampling_rate)
            except: # pylint: disable=W0702
                logging.debug('resampling noise %s failed. Loading random noise',  fnames[id])
                noise = np.random.randn(length).astype(np.float32) * 0.1
        elif sample_rate_in < target_sampling_rate:
            logging.debug('reading noise file %s sampling rate < 16000', fnames[rand_idx])
            noise = np.random.randn(length).astype(np.float32) * 0.1

        if len(noise) > length:
            start = np.random.randint(0, len(noise)-length)
            noise = noise[start : start+length]
        else:
            try:
                repeats = int(np.ceil(length / len(noise)))
            except: # pylint: disable=bare-except
                print(fnames[rand_idx])
                noise = np.random.randn(length)
            else:
                noise = np.tile(noise, repeats)
                noise = noise[:length]

    return noise
