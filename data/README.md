# data

Here I have some tools for downloading and preprocessing YouTube videos.
These came from a different repo -- so you might need to download different dependencies.

The rough workflow:
* First, get a bunch of YouTube IDs
* Download all of them somewhere (I used Google Cloud Storage)
* Use [process.py](process.py) to convert them into tfrecord format
* Then you can train the model.

A few pieces that could be useful:
* Our model for slightly improving the timing of YouTube ASR: [offset_model](offset_model)
* [process.py](process.py) converts audio into spectrograms
* [process.py](process.py) extracts frames from videos
* [process.py](process.py) also calls a lightweight MobileNet V2 CNN to remove videos whose images seem too similar
Also:
* [download_youtube.py](download_youtube.py) is a wrapper around YouTube-DL that you could use for downloading videos.