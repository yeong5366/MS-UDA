import yaml


class _Parser():

    def __init__(self):

        self.initialized = False

    def initialize(self, args, parser):

        with open(args.config, 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        MODEL = config['model_params']['name']

        DATASET = config['exp_params']['dataset']

        ROOT_DIR = config['exp_params']['root_dir']
        DATA_DIR = config['exp_params']['data_dir']
        NUM_CLASSES = config['exp_params']['num_classes']
        GPUS = config['exp_params']['gpus']
        INPUT_SIZE = config['exp_params']['input_size']
        NUM_LAYERS = config['exp_params']['num_layers']

        TRAIN_BATCH = config['train_params']['batch_size']
        MAX_EPOCH = config['train_params']['max_epoch']
        NUM_WORKERS = config['train_params']['num_workers']
        NORM = config['train_params']['norm']
        GAN_TYPE = config['train_params']['gan']
        ALPHA = config['train_params']['lambda']['alpha']
        BETA = config['train_params']['lambda']['beta']
        GAMMA = config['train_params']['lambda']['gamma']
        BASE_LR_G = config['train_params']['generator']['base_lr']
        MOMENTUM_G = config['train_params']['generator']['momentum']
        WEIGHT_DECAY_G = config['train_params']['generator']['weight_decay']
        BASE_LR_D = config['train_params']['discriminator']['base_lr']
        BASE_LR_Dec = config['train_params']['decoder']['base_lr']
        MOMENTUM_Dec = config['train_params']['decoder']['momentum']
        WEIGHT_DECAY_Dec = config['train_params']['decoder']['weight_decay']
        POWER = config['train_params']['power']

        TEST_BATCH = config['test_params']['batch_size']

        LOG_DIR = config['logging_params']['log_dir']
        CHECK_DIR = config['logging_params']['save_dir']
        TENSORBOARD = config['logging_params']['tensorboard']

        INIT_TYPE = config['init']['init_type']
        INIT_GAIN = config['init']['init_gain']

        parser.add_argument('--dataset', type=str, default = DATASET)
        parser.add_argument('--root_dir', type=str, default=ROOT_DIR)
        parser.add_argument('--data_dir', type=str, default=DATA_DIR)
        parser.add_argument('--num_classes', type=int, default=NUM_CLASSES)
        parser.add_argument('--train_batch', type=int, default=TRAIN_BATCH)
        parser.add_argument('--max_epoch', type=int, default=MAX_EPOCH)
        parser.add_argument('--alpha', default=ALPHA)
        parser.add_argument('--beta', default=BETA)
        parser.add_argument('--gamma', default=GAMMA)
        parser.add_argument('--base_lr_G', default=BASE_LR_G)
        parser.add_argument('--momentum_G', default=MOMENTUM_G)
        parser.add_argument('--weight_decay_G', default=WEIGHT_DECAY_G)
        parser.add_argument('--base_lr_D', default=BASE_LR_D)
        parser.add_argument('--base_lr_Dec', default=BASE_LR_Dec)
        parser.add_argument('--momentum_Dec', default=MOMENTUM_Dec)
        parser.add_argument('--weight_decay_Dec', default=WEIGHT_DECAY_Dec)

        parser.add_argument('--norm', type=str, default=NORM)
        parser.add_argument('--gan_type', type=str, default=GAN_TYPE)
        parser.add_argument('--test_batch', type=int, default=TEST_BATCH)
        parser.add_argument('--gpus', type=str, default=GPUS)
        parser.add_argument('--input_size', default=INPUT_SIZE)
        parser.add_argument('--num_layers', type=int, default=NUM_LAYERS)

        parser.add_argument('--check_dir', default=CHECK_DIR)
        parser.add_argument('--log_dir', default=LOG_DIR)
        parser.add_argument('--tensorboard', default=TENSORBOARD)
        parser.add_argument('--init_type', default=INIT_TYPE, help='xavier|normal|kaiming')
        parser.add_argument('--model', default=MODEL)
        parser.add_argument('--num_workers', default=NUM_WORKERS)
        parser.add_argument('--init_gain', default=INIT_GAIN)
        parser.add_argument('--power', default=POWER)

        self.initialized = True
        return parser

    def gather_option(self, args, parser):

        if not self.initialized:
            parser = self.initialize(args, parser)

        args = parser.parse_args()

        str_ids = args.gpus.split(',')
        args.gpus = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                args.gpus.append(id)

        self.option = args

        return self.option