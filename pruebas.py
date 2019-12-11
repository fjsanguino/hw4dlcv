from reader import getVideoList, readShortVideo
import torch

print("Working on getting videos")
dic = getVideoList('hw4_data/TrimmedVideos/label/gt_valid.csv')

video_idx = dic.get("Video_index")[0]

print(len(dic))
#for i in range(len(dic.get("Video_index"))):
index = []
fram = []
#for i in range(len(dic.get('Video_index'))):
    #print("working on video", i)
video = {}
for x, y in dic.items():
    video[x] = y[0]

frame = readShortVideo('hw4_data/TrimmedVideos/video/valid', video.get('Video_category'), video.get('Video_name'))
print(len(frame))
frame_res = torch.from_numpy(frame)
frame_res.resize_(12, 240, 240, 3)
index.append(video.get('Video_index'))
fram.append(readShortVideo('hw4_data/TrimmedVideos/video/valid', video.get('Video_category'), video.get('Video_name')))

frames = {
    'Video_index': index,
    'frames': fram
          }

exit(0)