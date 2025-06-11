import argparse
import time
from recbole.quick_start import run_recbole


if __name__ == '__main__':

    begin = time.time()
    parameter_dict = {
        'neg_sampling': None,

    }
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='DIFF', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='Amazon_Beauty', help='name of datasets')
    parser.add_argument('--config_files', type=str, default='configs/Amazon_Beauty_diff.yaml', help='config files')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--n_layers', type=int)
    parser.add_argument('--n_heads', type=int)
    parser.add_argument('--pooling_mode', type=str, default='sum')
    parser.add_argument('--tau', type=float, default=1)
    parser.add_argument('--c', type=int, default=5)
    parser.add_argument('--mask_ratio', type=float, default=0.2)

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_result = run_recbole(model=args.model, dataset=args.dataset, config_file_list=config_file_list, config_dict=parameter_dict)
    end = time.time()
    print(end-begin)
