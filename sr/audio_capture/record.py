import os.path

import pyaudio
import scipy
import time
import wave
import numpy as np


def decode_audio_stream(data, dtype=np.int16):
    return np.frombuffer(data, dtype).tolist()


class AudioFrame:
    def __init__(self, data, is_speech=False):
        self.data = data
        self.is_speech = is_speech
        # energy of the signal
        self.energy = 0
        # signal level
        self.level = 0

    def calc_energy(self):
        n = len(self.data)
        self.data = np.asarray(self.data)
        sum = np.sum(self.data ** 2)
        if sum <= 1:
            self.energy = 0
        else:
            self.energy = 10 * np.log10(sum)
        return self.energy


class AudioRecorder:
    def __init__(self, config=None):
        """
        :param config: dictionary containing configuration of the AudioRecorder.
                Available configurations are:
                - 'sample rate': sample rate of the recorded audio, defaults to 16000 Hz
                - 'format': format of each sample, defaults to pyaudio.paInt16
                - 'chunk size': chunk size in pyaudio, defaults to 1024
                - 'channel count': number of channels, defaults to 1
                - 'forget factor': forget factor in endpoint detection algorithm, defaults to 100,
                - 'max record time': max record time even if the user is still talking, in seconds,
                    defaults to 1000s
                - 'frame time': length of each frame in seconds, defaults to 0.02s
                - 'frame stride': length of each step between two frames, in seconds, defaults to 0.01s
                - 'adjustment': adjustment factor, defaults to 0.01
                - 'onset threshold': threshold for detecting speech start, defaults to 3
                - 'offset threshold': threshold for detecting speech end, defaults to 0.2
                - 'silence threshold': max length of silence before stop recording, in microseconds, defaults to 500ms
                - 'speech threshold': max length of speech before speech is recognized, in microseconds,
                    defaults to 250ms
                - 'start boundary': boundary added to the beginning of recorded speech, in ms, defaults to 200ms
                - 'end boundary': boundary added to the end of recorded speech, in ms, defaults to 0ms
        """
        if config is None:
            config = {
                'sample rate': 8000,
                'format': pyaudio.paInt16,
                'chunk size': 1024,
                'channel count': 1,
                'forget factor': 1,
                'max record time': 1000,
                'frame time': 0.02,  # in seconds
                'frame stride': 0.01,  # in seconds
                'adjustment': 0.01,
                'onset threshold': 3,
                'offset threshold': 0.2,
                'silence threshold': 500,  # in ms
                'speech threshold': 250,  # in ms
                'start boundary': 200,  # in ms
                'end boundary': 0,  # in ms
            }
        self.config = config
        # calculate how many samples there are in a frame which is
        # config['frame time'] long
        self.config['samples per frame'] = int(self.config['frame time'] * self.config['sample rate'])

        # calculate stride and frame width in samples
        self.config['frame stride'] = int(self.config['frame stride'] * self.config['sample rate'])

        # calculate the frame count of silence/speech threshold and update it
        # in place
        self.config['silence threshold'] = int(self.config['silence threshold'] * self.config['sample rate'] / (
                1000 * self.config['frame stride']))
        self.config['speech threshold'] = int(self.config['speech threshold'] * self.config['sample rate'] / (
                1000 * self.config['frame stride']))
        # calculate how many sample the boundary should be
        self.config['start boundary'] = int(self.config['start boundary'] / 1000 * self.config['sample rate'])
        self.config['end boundary'] = int(self.config['end boundary'] / 1000 * self.config['sample rate'])

        self.frames = []
        self.samples = []
        self.audio_driver = pyaudio.PyAudio()

        self.started_speech = False

        # cache data across frames
        self.cache_data = {
            'background': 0, 'silence time': 0, 'speech time': 0, 'frame count': 0,
            'boundary sample count': 0
        }
        self.curr_delay = 0

        self.speech_start_index = 0
        self.speech_end_index = 0

        # if True, the recorder wait for self.config['end boundary'] samples and stop recording
        self.should_end_recording = False
        # debug information
        self.levels = []
        self.backgrounds = []
        self.final_levels = []

    def record_callback(self, in_data, frame_count, time_info, status):
        decoded = decode_audio_stream(in_data)
        self.samples += decoded
        new_frame_count = 0

        # end recording
        if self.should_end_recording:
            self.cache_data[
                'boundary sample count'] += frame_count  # frame_count here is actually sample count
            # (it's from the pyaudio callback)
            if self.cache_data['boundary sample count'] >= self.config['end boundary']:
                return in_data, pyaudio.paComplete
            else:
                # skip code below because we don't need to clasify anything
                return in_data, pyaudio.paContinue

        if self.cache_data['frame count'] == 0:
            f = AudioFrame(decoded)
            self.frames.append(f)
            new_frame_count = 1
        else:
            stride = self.config['frame stride']
            width = self.config['samples per frame']
            new_frame_count = int(width / stride)

            new_frame_start = self.cache_data['frame count'] * stride

            # append the new frames, and classify them
            for i in range(0, new_frame_count):
                s = int(new_frame_start + i * stride)
                f = AudioFrame(self.samples[s: s + width])
                self.frames.append(f)

                # classify the new frames
                is_speech = self.classify_frame(self.cache_data['frame count'] + i)
                # update how long last silence is
                if is_speech:
                    self.cache_data['speech time'] += 1
                    self.cache_data['silence time'] = 0
                else:
                    self.cache_data['silence time'] += 1
                    self.cache_data['speech time'] = 0

                if self.cache_data['speech time'] > self.config['speech threshold'] and not self.started_speech:
                    self.cache_data['silence time'] = 0
                    self.started_speech = True
                    self.speech_start_index = s
                    print('detected speech')
                elif self.cache_data['silence time'] > self.config['silence threshold'] and self.started_speech:
                    self.started_speech = False
                    self.speech_end_index = s + width
                    print('speech ended')
                    self.should_end_recording = True
                    return in_data, pyaudio.paContinue

        # increase total frame count
        self.cache_data['frame count'] += new_frame_count
        # continue recording
        return in_data, pyaudio.paContinue

    def classify_frame(self, frame_index):
        frame = self.frames[frame_index]
        is_speech = False
        current_energy = frame.calc_energy()
        # if this is before the first 10th frames, current signal level is the
        # energy
        if frame_index <= 10:
            frame.level = current_energy
        else:
            last_frame = self.frames[frame_index - 1]
            frame.level = (last_frame.level + (self.config['forget factor'] * current_energy)) / (
                    self.config['forget factor'] + 1)
            is_speech = last_frame.is_speech

        # if this is the 10th frames, the background
        # level is the average energy of these frames
        if frame_index < 10:
            return False
        elif frame_index == 10:
            for f in self.frames:
                self.cache_data['background'] += f.energy
            self.cache_data['background'] /= 10
        else:  # otherwise calculate the background signal level and the current signal level
            self.cache_data['background'] += (current_energy - self.cache_data['background']) * self.config[
                'adjustment']

        if frame.level < self.cache_data['background']:
            frame.level = self.cache_data['background']
        elif frame.level - self.cache_data['background'] > self.config['onset threshold']:
            frame.is_speech = True
            is_speech = True
        elif frame.level - self.cache_data['background'] < self.config['offset threshold']:
            frame.is_speech = False
            is_speech = False
        else:
            frame.is_speech = is_speech

        self.final_levels.append(frame.level - self.cache_data['background'])
        self.levels.append(frame.level)
        self.backgrounds.append(self.cache_data['background'])

        return is_speech

    def start_recording(self, visualize=False):
        stream = self.audio_driver.open(format=self.config['format'],
                                        channels=self.config['channel count'],
                                        rate=self.config['sample rate'],
                                        input=True,
                                        output=False,
                                        frames_per_buffer=self.config['samples per frame'],
                                        stream_callback=self.record_callback
                                        )
        stream.start_stream()
        # record for 5 seconds
        start_time = time.time()
        while stream.is_active():
            # get time lasted since recording
            end_time = time.time()
            time.sleep(0.1)
            # if timeout, stop recording
            if end_time - start_time > self.config['max record time']:
                break

        stream.stop_stream()
        stream.close()
        self.audio_driver.terminate()

    def get_samples(self, dtype=np.int16):
        s = self.speech_start_index - self.config['start boundary']
        if s < 0:
            s = 0
        e = self.speech_end_index
        return np.array(self.samples[s:e + 1]).astype(dtype).copy()

    def write_to_wav_file(self, file_name):
        wf = wave.open(file_name, 'wb')
        wf.setnchannels(self.config['channel count'])
        wf.setsampwidth(self.audio_driver.get_sample_size(self.config['format']))
        wf.setframerate(self.config['sample rate'])

        # get indices of start and end of all samples
        s = self.speech_start_index - self.config['start boundary']
        if s < 0:
            s = 0
        e = self.speech_end_index

        wf.setnframes(e - s + 1)
        wf.writeframes(np.array(self.samples[s:e + 1]).astype(np.int16).tobytes())
        wf.close()


def record(file=None):
    # make sure the directory that contains the output file exists
    os.makedirs(os.path.dirname(file), exist_ok=True)

    # record
    ar = AudioRecorder()
    ar.start_recording()
    if file:
        ar.write_to_wav_file(file)
    return ar
