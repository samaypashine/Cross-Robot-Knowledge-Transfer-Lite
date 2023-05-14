# Author: Gyan Tatiya

import argparse
import os
import subprocess
import time

import librosa

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make videos from images and audio file. This script keeps making '
                                                 'videos in a while loop, so terminate when needed.')
    parser.add_argument('-dataset',
                        choices=['Tool_Dataset', 'Tool_Dataset_2Tools_2Contents'],
                        # required=True,
                        default='Tool_Dataset',
                        help='dataset name')
    args = parser.parse_args()

    sensor_data_path = r'data' + os.sep + args.dataset

    while True:
        for root, subdirs, files in os.walk(sensor_data_path):
            print('\nroot: ', root)

            if 'camera_rgb_image' in root:

                root_list = root.split(os.sep)
                behavior_dir = os.sep.join(root_list[:-1])

                audio_file = behavior_dir + os.sep + 'audio' + os.sep + 'audio.wav'
                # print("audio_file: ", audio_file)

                video_output_file = os.sep.join(root_list[:-1]) + os.sep + 'video.mp4'

                if os.path.exists(video_output_file) or (not os.path.exists(audio_file)):
                    print('Video already exists or audio not available yet!')
                    continue

                audio_time_series, sampling_rate = librosa.load(audio_file, sr=16000)
                audio_length = len(audio_time_series) / sampling_rate
                # print("audio_length: ", audio_length)

                frequency = len(files) / audio_length
                # print("frequency: ", frequency)

                cmd = "ffmpeg -r " + str(frequency) + " -pattern_type glob -i '" + root + os.sep + "*.jpg' -i '" \
                      + audio_file + "' -strict experimental -async 1 '" + video_output_file + "' -y"
                print('cmd: ', cmd)
                p = subprocess.Popen(cmd, stdin=subprocess.PIPE, shell=True)
                time.sleep(30)
