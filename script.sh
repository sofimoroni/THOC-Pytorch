echo "gpu $1 $2 $3 $4"
CUDA_VISIBLE_DEVICES=$1 python main.py --hidden_dim 128 --L2_reg --epochs 1 --dataset NeurIPS-TS-MUL &
CUDA_VISIBLE_DEVICES=$2 python main.py --hidden_dim 128 --epochs 1 --dataset SWaT &
CUDA_VISIBLE_DEVICES=$3 python main.py --hidden_dim 128 --epochs 1 --dataset SMAP &
CUDA_VISIBLE_DEVICES=$4 python main.py --hidden_dim 128 --epochs 1 --dataset MSL &

# My data
python main.py --hidden_dim 128 --epochs 1 --dataset MY_DATA --window_size 120 --stride 1 --lr 1e-04 --L2_reg 4
