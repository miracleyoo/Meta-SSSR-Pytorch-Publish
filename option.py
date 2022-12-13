import argparse
import template
import platform
import getpass

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=0,
                    help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')

# Data specifications
parser.add_argument('--dir_data', type=str, default=r'L:\Satellite\prepared',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='NTIRE',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='NTIRE_VAL',
                    help='test dataset name')
parser.add_argument('--data_range', type=str, default='1-800/801-810',
                    help='train/test data range')
parser.add_argument('--ext', type=str, default='sep',
                    help='A flag which defines the dataset loading strategy.\
                    Choices: sep, img, bin. You can add a _reset to reset bin\
                    files generated.')
parser.add_argument('--scale', type=int, default=3,
                    help='super resolution scale')
parser.add_argument('--scale_test', type=int, default=3,
                    help='super resolution scale for test')
parser.add_argument('--patch_size', type=int, default=40,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')

# HSI Data specification
parser.add_argument('--n_hc', type=int, default=10,
                    help='number of hsi channels to use, used in edsrfuse and rdnfuse.')
parser.add_argument('--hsi_mean', type=float, default=0.45,
                    help='default mean value for hsi channels.')
parser.add_argument('--hsi_std', type=float, default=1,
                    help='default standard deviation value for hsi channels.')
parser.add_argument('--hsi_range', type=int, default=255,
                    help='maximum value of HSI')
parser.add_argument('--cwl_max', type=int, default=1000,
                    help='maximum value of central wavelength supported.')
parser.add_argument('--cwl_min', type=int, default=100,
                    help='minimum value of central wavelength supported.')
parser.add_argument('--cwl_center', type=int, default=550,
                    help='center value of central wavelength supported.')
parser.add_argument('--max_in_channel', type=int, default=9,
                    help='minimum value of central wavelength supported.')
parser.add_argument('--min_in_channel', type=int, default=5,
                    help='minimum value of central wavelength supported.')

parser.add_argument('--hsi_avg_bw', type=int, default=10,
                    help='Default mean bandwidth for each input channel in hsi images.')
parser.add_argument('--rgb_avg_bw', type=int, default=100,
                    help='Default mean bandwidth for each input channel in rgb images.')
parser.add_argument('--jitter', action='store_true',
                    help='Add jitter to the channel selection when evenly sample.')


parser.add_argument('--data_config', default="config.json",
                    help='wavelength specification file name in the dataset root.')
# TODO: Debug all, rgb, set mode.
parser.add_argument('--wl_out_type', default="same",
                    choices=('same', 'all', 'rgb', 'set',
                             'max', 'rand', 'comp'),
                    help='Output wavelength type. (same|all|rgb|set|max|rand|comp)')
parser.add_argument('--wl_in_type', default="even",
                    choices=('even', 'all', 'rand',
                             'mid2side', 'start2end', 'set'),
                    help='Output wavelength type. (even|all|rand|mid2side|start2end|set)')
parser.add_argument('--in_bands_num', type=int, default=21,
                    help='Band number used in the input HSI.')


# MetaFMRDN specifications
parser.add_argument('--ca_type', default="none",
                    choices=('eca', 'se', 'san', 'se_res', 'none'),
                    help='Output wavelength type. (eca|se|san|se_res|none)')
parser.add_argument('--up_type', default="1d",
                    choices=('1d', '2d'),
                    help='Upsampling method. (1d|2d)')
parser.add_argument('--cd_channel', type=int, default=0,
                    help='Condensed feature map channel number in 2d meta upsample module.')
parser.add_argument('--skip_bw', action='store_true',
                    help='Do not use bandwidth in meta block')
parser.add_argument('--tail_type', default="multi",
                    choices=('parallel', 'multi', 'rgb', 'hsi', 'pmulti'),
                    help='Tail type. (parallel|multi|pmulti|rgb|hsi)')
parser.add_argument('--avg_cl', action='store_true',
                    help='Use the mean value of all convlstm/biconvlstm cell output instead of the last h.')


# Model specifications
parser.add_argument('--model', default='metafrdn',
                    help='model name')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')

parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--dilation', action='store_true',
                    help='use dilated convolution')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Option for Residual dense network (RDN)
parser.add_argument('--H0', type=int, default=32,
                    help='default number of filters. (Use in Header)')
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--G_hidden', type=int, default=64,
                    help='Inner conv 2d layer output channels in RDB.')
parser.add_argument('--rdb_conv_num', type=int, default=8,
                    help='Number of conv 2d layers in each RDB.')
parser.add_argument('--head_blocks', type=int, default=8,
                    help='Number of dense blocks used in head part.')
parser.add_argument('--body_blocks', type=int, default=8,
                    help='Number of dense blocks used in body part.')
parser.add_argument('--tail_blocks', type=int, default=4,
                    help='Number of dense blocks used in tail part.')

parser.add_argument('--WLkSize1', type=int, default=1,
                    help='default kernel size. (Use in wavelength->weight network 1)')
parser.add_argument('--WLkSize2', type=int, default=1,
                    help='default kernel size. (Use in wavelength->weight network 2)')
parser.add_argument('--old_shift', action='store_true',
                    help='Use the old version of input/output mean shift.')
parser.add_argument('--head_type', type=str, default='meta',
                    choices=('meta', 'add', 'cat'),
                    help='FP precision for test (meta | add | cat)')
parser.add_argument('--mix_type', type=str, default='average',
                    choices=('average', 'max', 'convlstm', 'biconvlstm'),
                    help='FP precision for test (average | convlstm)')
parser.add_argument('--cl_layers', type=int, default=3,
                    help='Number of ConvLSTM inner layers.')

parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--wlwp_epoch', type=int, default=1000,
                    help='number of epochs to stop training weight prediction net.')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--no_test', action='store_true',
                    help='Do not test the model, train only.')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')
parser.add_argument('--double_train', action='store_true',
                    help='Go through the rgb->hsi and hsi->rgb training process.')
parser.add_argument('--plot', action='store_true',
                    help='Plot PSNR and loss while training.')
parser.add_argument('--tbn', type=int, default=0,
                    help='How many input bands setting to test each time during training.')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=50,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='resume from the snapshot, and the start_epoch')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*L1',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e5',
                    help='skipping batch that has large error')
parser.add_argument('--rgb_loss_weight', type=float, default='1',
                    help='The RGB loss weight when combined with HSI loss')

# Log specifications
parser.add_argument('--save', type=str, default='meta',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',
                    help='save output results')


# Test Cases
parser.add_argument('--test_case', type=str, default='',
                    help='Test case name.')


def init():
    args = parser.parse_args()
    template.set_template(args)
    return process_args(args)


def process_args(args):
    server_name = platform.node()
    if server_name == 'misaka':
        args.dir_data = '../satellite_pure'
    elif server_name.startswith('node') or server_name == 'gypsum':
        args.dir_data = './datasets'
    # if 'zhiyangxu' in getpass.getuser():
    #     args.dir_data = '/mnt/nfs/scratch1/zhongyangzha/Meta-FM-SR-Pytorch/datasets'

    # Parse the data_train and data_test
    if len(args.data_train.split('+')) > 1:
        args.data_train = list(
            map(lambda x: x.strip(), args.data_train.split('+')))

    if len(args.data_test.split('+')) > 1:
        args.data_test = list(
            map(lambda x: x.strip(), args.data_test.split('+')))

    if args.epochs == 0:
        args.epochs = 1e3

    # Bool compatibility
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    # Central wavelength for R,G,B, unit:nm
    args.cwl_rgb = [625, 545, 455]
    # Central wavelength for all visible spectrums with a step of 10nm, unit:nm
    args.cwl_hsi = list(range(400, 710, 10))
    # Central wavelength for specific user definition, unit:nm
    args.cwl_set = [625, 545, 455]

    args.avg_bw_set = [100, 100, 100]

    # Mean and standard deviation of rgb and hsi
    args.mean_rgb = [114, 111, 103]
    args.mean_hsi = [31, 31, 31, 32, 35, 39, 41, 42, 42, 42, 43, 43, 44, 46,
                     48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 48, 47, 47, 48, 47, 46, 48]
    args.mean_set = [114, 111, 103]

    args.std_rgb = [72, 69, 74]
    args.std_hsi = [39, 39, 39, 39, 43, 46, 48, 49, 49, 48, 48, 47, 47, 48,
                    49, 50, 50, 49, 49, 49, 50, 50, 50, 49, 49, 49, 50, 50, 48, 47, 46]
    args.std_set = [72, 69, 74]

    args.bands_in_set = [2, 5, 11, 20, 28]

    return args
