python main.py /cluster/work/erlandbo/imagenet/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclrall --rho 0.9 --no-single_s --checkpoint-dir all_wd1e6 --weight-decay 1e-6
python main.py /cluster/work/erlandbo/imagenet/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclr1 --rho 0.99 --single_s --checkpoint-dir one_wd1e6 --weight-decay 1e-6
python main.py /cluster/work/erlandbo/imagenet/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclrall --rho 0.9 --no-single_s --checkpoint-dir all_wd1e4 --weight-decay 1e-4
python main.py /cluster/work/erlandbo/imagenet/ --learning-rate-weights 1.2 --learning-rate-biases 1.2 --batch-size 256 --epochs 200 --method saclr1 --rho 0.99 --single_s --checkpoint-dir one_wd1e4 --weight-decay 1e-4

#python main.py /cluster/work/erlandbo/imagenet/ ---learning-rate-weights 2.4 ---learning-rate-biases 2.4 --batch-size 256 --epochs 200 --method saclrall --checkpoint-dir saclrall100_bt_default ----weight-decay 1e-6
