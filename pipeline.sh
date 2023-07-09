python selfplay_main.py --save-dir archive --model model/sl-model.bin --use-gpu true
python get_final_status.py
python train.py --rl true --kifu-dir archive

for i in `seq 1 100` ; do
    python selfplay_main.py --save-dir archive --model model/rl-model.bin --use-gpu true
    python get_final_status.py
    python train.py --rl true --kifu-dir archive
done
