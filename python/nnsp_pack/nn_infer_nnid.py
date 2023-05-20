""" test NNID """
import wave
import time
import numpy as np
from .nn_infer import NNInferClass
from .nn_activation import softmax
from .feature_module import display_stft

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
                sample_rate=self.params_audio['sample_rate'])

        return embds[-1]
