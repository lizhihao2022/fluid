from .fno_2d import fno_burgers
from .unet_2d import unet_2d_burgers
from .pino import pino_burgers
from .unet_1d import unet_1d_burgers


def burgers_procedure(args):
    if args['model'] == 'FNO2d':
        fno_burgers(args)
    elif args['model'] == 'UNet2d':
        unet_2d_burgers(args)
    elif args['model'] == 'UNet1d':
        unet_1d_burgers(args)
    elif args['model'] == 'PINO':
        pino_burgers(args)
    else:
        raise NotImplementedError
