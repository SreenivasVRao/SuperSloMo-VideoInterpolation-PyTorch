"""
Generate train/val sequences.
25 frames per clip. 9 frame overlap between adjacent clips.
25 frames: I_0,< 7 frames> , I_1, <7 frames>....I_3
"""

import os
import glob
import numpy as np
import pickle
np.random.seed(42)

# step can be negative indicating there is overlap between two clips
def process_single_dir(im_dir, num_frames_per_clip=9, step=10):
    clips = []
    im_names = sorted(glob.glob(os.path.join(im_dir, '*.png')))
    start_idx = 0
    while start_idx < len(im_names):
        end_idx = start_idx + num_frames_per_clip
        if end_idx >=len(im_names):
            break # guarantee at least num_frames_per_clip.

        clips.append(im_names[start_idx : end_idx])
        # end_idx = min(start_idx + num_frames_per_clip, len(im_names))
        # if (end_idx + step > len(im_names) or
        #       end_idx + step + num_frames_per_clip > len(im_names)):
        #     end_idx = len(im_names)

        # if (end_idx - start_idx >= num_frames_per_clip or
        #         end_idx - start_idx >= 9):
        #     clips.append(im_names[start_idx : end_idx])
        start_idx = end_idx + step
    return clips

def process_data_dir(data_dir, num_frames_per_clip, step):
    seqs = sorted(glob.glob(os.path.join(data_dir, '*')))
    print('There are %d sequences in %s' % (len(seqs), data_dir))

    # np.random.shuffle(seqs)
    # num_train = int(len(seqs) * 0.9)
    # train_seqs = seqs[:num_train]
    # val_seqs = seqs[num_train:]

    with open("/home/sreenivasv/original.pkl", "rb") as f:
        train_clips = pickle.load(f)

    with open("/home/sreenivasv/original_val.pkl", "rb") as f:
        val_clips = pickle.load(f)

    train_seqs = [os.path.join(data_dir, clip) for clip in train_clips]
    val_seqs = [os.path.join(data_dir, clip) for clip in val_clips]
    print('%d seqs for training and %d for validation.' % (len(train_seqs), len(val_seqs)))
    train_clips = []
    for s in train_seqs:
        clips = process_single_dir(s, num_frames_per_clip, step)
        train_clips.extend(clips)
    print('%d clips found for training.' % len(train_clips))

    val_clips = []
    for s in val_seqs:
        clips = process_single_dir(s, num_frames_per_clip, step)
        val_clips.extend(clips)
    print('%d clips found for validation.' % len(val_clips))
    return train_clips, val_clips

if __name__ == '__main__':
    data_dirs = ['/mnt/nfs/work1/elm/hzjiang/Data/VideoInterpolation/Adobe240fps/Clips']
    # data_dirs = ['/home/huaizuj/Data/Adobe240fps/Clips']

    num_frames_per_clip = 57
    step = -40

    all_train_clips = []
    all_val_clips = []
    for dd in data_dirs:
        train_clips, val_clips = process_data_dir(dd, num_frames_per_clip, step)
        all_train_clips.extend(train_clips)
        all_val_clips.extend(val_clips)
    print('%d clips for training and %d for validation.' % (len(all_train_clips), len(all_val_clips)))

    with open('train_clips_video_interp_all.txt', 'w') as f:
        f.write('%d\n' % len(all_train_clips))
        for clp in all_train_clips:
            f.write('%d\n' % len(clp))
            for n in clp:
                f.write('%s\n' % n)

    with open('val_clips_video_interp_all.txt', 'w') as f:
        
        for clp in all_val_clips:
            f.write('%d\n' % len(clp))
            for n in clp:
                f.write('%s\n' % n)

    # im_dir = '/home/huaizuj/Data/Adobe240fps/Clips/clip_00001'
    # clips = process_single_dir(im_dir, 30)
    # print(len(clips))
    # for clp in clips:
    #     print(clp[0])
    #     print(clp[-1])
    #     print('-------------------------------')
