import argparse

from main import MainModule
from final_module import train, test

def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_x', default='', type=str,
                        help='path where input for nn should be saved')
    parser.add_argument('--path_y', default='', type=str,
                        help='path where true actions should be saved')
    parser.add_argument('--path_valid_x', default='', type=str,
                        help='path to dataset')
    parser.add_argument('--path_valid_y', default='', type=str,
                        help='path to true actions for dataset')
    parser.add_argument('--pan_file', default='', type=str,
                        help='output produced by pan for training dataset')
    parser.add_argument('--pan_file_validation', default='', type=str,
                        help='output produced by pan for training dataset')
    parser.add_argument('--dataset_train', default='', type=str)
    parser.add_argument('--dataset_valid', default='', type=str)
    parser.add_argument('--train', default=False, action='store_true',
                        help='starts training')
    parser.add_argument('--test', default=False, action='store_true',
                        help='starts testing')
    parser.add_argument('--num_classes', default='', type=str)
    parser.add_argument('--model', default='pretrained/model', type=str,
                        help='path to model')
    args = parser.parse_args(argv)
    return args

if __name__ == "__main__":
    args = parse_args()
    mainModule = MainModule()
    if args.train:
        mainModule.evaluate(args.pan_file, args.dataset_train, args.path_x, args.path_y)
        mainModule.evaluate(args.pan_file_validation, args.dataset_valid, args.path_valid_x, args.path_valid_y)
        train(args.path_x, args.path_y, args.path_valid_x, args.path_valid_y, args.model, int(args.num_classes))
    if args.test:
        mainModule.evaluate(args.pan_file_validation, args.dataset_valid, args.path_valid_x, args.path_valid_y)
        test(args.path_valid_x, args.path_valid_y, args.model, int(args.num_classes))
