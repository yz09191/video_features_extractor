import os
import cv2
import shutil
import imageio
import numpy as np


def sample_frames(video_path, max_frames):
    '''
    对视频帧进行采样，减少计算量。等间隔地取max_frames帧
    '''
    try:
        cap = cv2.VideoCapture(video_path)
    except:
        print('Can not open %s.' % video_path)
        pass
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        # 把BGR的图片转换成RGB的图片，因为之后的模型用的是RGB格式
        if ret is False:
            break
        # frame = frame[:, :, ::-1]  #::-1翻转
        frames.append(frame)
        frame_count += 1
    indices = np.linspace(8, frame_count - 7, max_frames, endpoint=False, dtype=int)
    frames = np.array(frames)
    frame_list = frames[indices]
    clip_list = []
    for index in indices:
        clip_list.append(frames[index - 8: index + 8])
    clip_list = np.array(clip_list)
    return frame_list, clip_list


def extract_frames(dirname, output_path, video_name, frame_num):
    """Extract frames in a video. """
    video_path = os.path.join(dirname, video_name)
    frames, clips = sample_frames(video_path, frame_num)

    # extract frames
    frame_path = os.path.join(output_path, '%s_Frames' % video_name[:-4])
    if os.path.exists(frame_path):
        print(" cleanup: " + frame_path + "/")
        shutil.rmtree(frame_path)
    os.makedirs(frame_path)
    for i, frame in enumerate(frames):
        imageio.imwrite(os.path.join(frame_path, str(i) + '.jpg'), frame)
    assert len(os.listdir(frame_path)) == frame_num, 'Wrong frame number...'

    # extract clips
    clips_path = os.path.join(output_path, '%s_Clips' % video_name[:-4])
    if os.path.exists(clips_path):
        print(" cleanup: " + clips_path + "/")
        shutil.rmtree(clips_path)
    os.makedirs(clips_path)
    for i, clip in enumerate(clips):
        clip_path = os.path.join(clips_path, 'clip' + str(i))
        os.makedirs(clip_path)
        for j, img in enumerate(clip):
            imageio.imwrite(os.path.join(clip_path, str(j) + '.jpg'), img)


def crop_frames(frame_path, target_height=224, target_width=224):
    """
    resize and crop images
    """
    images = os.listdir(frame_path)
    for image in images:
        image_path = os.path.join(frame_path, image)
        image = cv2.imread(image_path)

        if len(image.shape) == 2:
            # 把单通道的灰度图复制三遍变成三通道的图片
            image = np.tile(image[:, :, None], 3)
        elif len(image.shape) == 4:
            image = image[:, :, :, 0]

        height, width, channels = image.shape
        if height == width:
            resized_image = cv2.resize(image, (target_height, target_width))
        elif height < width:
            resized_image = cv2.resize(image, (int(width * target_height / height), target_height))
            # print(resized_image.shape)
            cropping_length = int((resized_image.shape[1] - target_width) / 2)
            # print(cropping_length)
            resized_image = resized_image[:, cropping_length: target_width + cropping_length]
            # print(resized_image.shape)
        else:
            resized_image = cv2.resize(image, (target_width, int(height * target_width / width)))
            cropping_length = int((resized_image.shape[0] - target_height) / 2)
            resized_image = resized_image[cropping_length: target_height + cropping_length, :]
        cv2.imwrite(image_path, resized_image)


if __name__ == '__main__':
    path = 'F:\\dataset\\MSR-VTT\\Videos'
    video = 'video11.mp4'
    outpath = os.path.join('F:/PycharmProjects/video_features_extractor')
    extract_frames(path, outpath, video, 26)
    # crop_frames(outpath)
