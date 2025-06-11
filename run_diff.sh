gpu_id=0

# Yelp
python run_model.py --gpu_id=${gpu_id} --model=DIFF --dataset="yelp" --config_files="configs/yelp_diff.yaml" --fusion_type='gate' --alpha=0.3 --lambda=20

# Beauty
python run_model.py --gpu_id=${gpu_id} --model=DIFF --dataset="Amazon_Beauty" --config_files="configs/Amazon_Beauty_diff.yaml" --fusion_type='concat' --alpha=0.3 --lambda=100

# Sports
python run_model.py --gpu_id=${gpu_id} --model=DIFF --dataset="Amazon_Sports_and_Outdoors" --config_files="configs/Amazon_Sports_and_Outdoors_diff.yaml" --fusion_type='concat' --alpha=0.3 --lambda=100

# Toys
python run_model.py --gpu_id=${gpu_id} --model=DIFF --dataset="Amazon_Toys_and_Games" --config_files="configs/Amazon_Toys_and_Games_diff.yaml" --fusion_type='concat' --alpha=0.5 --lambda=100