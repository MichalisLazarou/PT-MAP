#python semisupervised_test.py --dataset miniImagenet --n_shots 1 --alpha 0.8 --K 20 --n_test_runs 500 --best_samples 3 --n_unlabelled 30 --lr 0.000001 --use_pt pt_transform
#python semisupervised_test.py --dataset miniImagenet --n_shots 5 --alpha 0.2 --K 30 --n_test_runs 500 --best_samples 5 --n_unlabelled 50 --lr 0.000001 --use_pt pt_transform

python test_standard.py --dataset tieredimagenet --n_shots 1 --alpha 0.8 --K 20 --n_test_runs 1000 --best_samples 3 --n_unlabelled 30 
python test_standard.py --dataset tieredimagenet --n_shots 5 --alpha 0.8 --K 20 --n_test_runs 1000 --best_samples 5 --n_unlabelled 50

python test_standard.py --dataset CUB --n_shots 1 --alpha 0.8 --K 20 --n_test_runs 1000 --best_samples 3 --n_unlabelled 30
#python semisupervised_test.py --dataset CUB --n_shots 5 --alpha 0.5 --K 25 --n_test_runs 1000 --best_samples 5 --n_unlabelled 50

python test_standard.py --dataset cifar --n_shots 1 --alpha 0.4 --K 20 --n_test_runs 1000 --best_samples 3 --n_unlabelled 30
python test_standard.py --dataset cifar --n_shots 5 --alpha 0.5 --K 25 --n_test_runs 1000 --best_samples 5 --n_unlabelled 50
