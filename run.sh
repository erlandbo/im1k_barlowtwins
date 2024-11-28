python main.py ~/Datasets/imagenet100/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclrall --checkpoint-dir saclrallim1k --weight-decay 1e-6
python main.py ~/Datasets/imagenet100/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclr1 --checkpoint-dir saclrallim1k --weight-decay 1e-6

#python main.py /cluster/work/erlandbo/imagenet/ ---learning-rate-weights 2.4 ---learning-rate-biases 2.4 --batch-size 256 --epochs 200 --method saclrall --checkpoint-dir saclrall100_bt_default ----weight-decay 1e-6
