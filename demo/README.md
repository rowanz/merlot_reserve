# THE DEMO

Here, `demo_video.py` is a python script for interactive Q/A on videos.

First download the video
```bash
pip install youtube-dl
youtube-dl -f "best[height<=480,ext=mp4]" https://www.youtube.com/watch?v=pmjPjZZRhNQ -o "%(id)s.%(ext)s"
```
Then run the demo! `ipython -i demo_video.py`

Check out [zero_shot_ek](zero_shot_ek)  and [zero_shot_qa](zero_shot_qa) for using it on QA tasks.