# TVQA

First, download the dataset --

Start off with something like
```bash
ls -ltr
total 175472548
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aa
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ab
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ac
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ad
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ae
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.af
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ag
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ah
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ai
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aj
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ak
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.al
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.am
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.an
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ao
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ap
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aq
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ar
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.as
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.at
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.au
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.av
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.aw
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ax
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ay
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.az
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.ba
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bb
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bc
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bd
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.be
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bf
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bg
-rw-rw-r-- 1 rowan rowan 4294967296 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bh
-rw-rw-r-- 1 rowan rowan  761207798 Nov 12  2018 tvqa_video_frames_fps3_hq.tar.gz.bi
-rw-rw-r-- 1 rowan rowan       2450 Nov 25  2018 tvqa_video_frames_fps3_hq.checksum.txt
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.aa
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ab
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ac
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ad
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ae
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.af
-rw-rw-r-- 1 rowan rowan 4294967296 Apr 29  2019 tvqa_audios.tar.gz.ag
-rw-rw-r-- 1 rowan rowan 2750112255 Apr 29  2019 tvqa_audios.tar.gz.ah
-rw-rw-r-- 1 rowan rown        448 Apr 29  2019 tvqa_audios.checksum.txt
```
Now extract the audio:
* `cat tvqa_audios.tar.gz.* | tar xz -C tvqa_frames`

Use `prep_data.py` to preprocess everything. 

Then, use `tvqa_finetune.py` to finetune the model and `submit_to_leaderboard.py` to submit results.

Something like (use the hyperparameters in the paper though, they differ between base and large).
```bash
python3 tvqa_finetune.py ../../pretrain/configs/large.yaml ${path_to_checkpoint} -lr=5e-6 -ne=3 -output_grid_h=18 -output_grid_w=32 -scan_minibatch
```