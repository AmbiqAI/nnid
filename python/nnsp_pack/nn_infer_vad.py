"""
VAD module including feature extraction and inference
"""
import wave
import time
import numpy as np
import matplotlib.pyplot as plt
from .nn_activation import softmax
from .nn_infer import NNInferClass
from .feature_module import display_stft

SHOW_HISTOGRAM  = False
NP_INFERENCE    = False

class VadClass(NNInferClass):
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
            thresh_prob=0.3,
            name_wavout=None,
            show_fig=False):
        """
        NN process for several frames
        """
        params_audio = self.params_audio
        start = 0
        count_conti = [0,0]
        start = [0,0]
        if name_wavout:
            file = wave.open(name_wavout, "wb")
            file.setnchannels(2)
            file.setsampwidth(2)
            file.setframerate(params_audio['sample_rate'])

        bks = int(len(data) / params_audio['hop'])
        feats = []
        specs = []
        triggers = data.copy()
        vad_triggers = np.zeros((bks,), dtype=int)
        probs = data.copy()
        stime = time.time()
        for i in range(bks):
            data_frame = data[i*params_audio['hop'] : (i+1) * params_audio['hop']]
            feat, spec = self.frame_proc_tf(data_frame, thresh_prob=thresh_prob)

            probs[i*params_audio['hop'] : (i+1) * params_audio['hop']] = self.vad_prob

            if self.cnt_vad_trigger[self.vad_trigger] >= 4:
                print(f'\rFrame {i}: trigger', end="")
                if self.cnt_vad_trigger[self.vad_trigger] == 4:
                    start[-1] = i
                triggers[i*params_audio['hop'] : (i+1) * params_audio['hop']] = 0.5
                vad_triggers[i] = 1
                count_conti[-1] += 1
                if count_conti[-1] == 180:
                    break
            else:
                print(f'\rFrame {i}:', end='')
                triggers[i*params_audio['hop'] : (i+1) * params_audio['hop']] = 0
                vad_triggers[i] = 0
                if count_conti[-1] > count_conti[0]:
                    count_conti[0] = count_conti[1]
                    start[0] = start[1]
                    start[-1] = 0
                    count_conti[-1] = 0
            feats += [feat]
            specs += [spec]
            self.count_run = (self.count_run + 1) % self.num_dnsampl
        etime = time.time()
        print("")
        print(f"ave {(etime-stime)/bks * 1000:2f} ms/inf")
        feats = np.array(feats)
        specs = np.array(specs)

        zeros = np.zeros((params_audio['hop']*2,))
        data = np.concatenate((data,zeros))
        probs = np.concatenate((zeros,probs))
        out = np.empty(
                (data.size + triggers.size + params_audio['hop'] * 2,),
                dtype=data.dtype)
        out[0::2] = data
        out[1::2] = 0.5 * (probs >= thresh_prob).astype(int)
        out = np.floor(out * 2**15).astype(np.int16)
        if name_wavout:
            file.writeframes(out.tobytes())
            file.close()
        if show_fig:
            display_stft(
                data, specs.T,
                feats.T, feats.T,
                sample_rate=self.params_audio['sample_rate'])

            plt.figure(2)
            ax_handle = plt.subplot(3,1,1)
            ax_handle.plot(data, linewidth=0.2)
            plt.ylim([-1,1])
            ax_handle = plt.subplot(3,1,2)
            ax_handle.plot(probs)
            plt.ylim([0,1])
            ax_handle = plt.subplot(3,1,3)
            ax_handle.plot(triggers)
            plt.show()

        return vad_triggers, start[-1]
