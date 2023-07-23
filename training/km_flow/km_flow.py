from .pino import pino_km_flow


def km_flow_procedure(args):
    if args['model'] == 'PINO':
        pino_km_flow(args)
    else:
        raise NotImplementedError