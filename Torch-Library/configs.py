import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument("--bool_arg", type=bool, default=True, help="")
    # parser.add_argument('--flag_arg', action='store_true', help="")
    # parser.add_argument('--int_arg', type=int, default=0, help="")
    # parser.add_argument('--float_arg', type=float, default=0.0, help="")
    # parser.add_argument('--str_arg', type=str, default='', choices=['', '', ''], help="")

    args = parser.parse_args()
    return args
