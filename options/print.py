import os
from datetime import datetime

def print_options(args, parser, print_set=True):
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    expr_dir = os.path.join(args.root_dir, args.model+args.log_dir)
    if not os.path.exists(expr_dir):
        os.makedirs(expr_dir)
    if print_set:
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(datetime.today()))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')