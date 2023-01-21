import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument("--model", type=str, default="LSTM",
                           help="Model name")
    model_arg.add_argument("--nlayers", type=int, default=3,
                           help="Number of LSTM layers")
    model_arg.add_argument("--nhid", type=int, default=768,
                           help="Hidden size")
    model_arg.add_argument("--dropout", type=float, default=0.2,
                           help="dropout between LSTM layers except for last")
    # Transformer
    transformer_arg = parser.add_argument_group('Transformer')
    transformer_arg.add_argument("--enc_layers", type=int, default=4,
                           help="Encoder Layers")
    transformer_arg.add_argument("--dec_layers", type=int, default=4,
                           help="Decoder Layers")
    transformer_arg.add_argument("--heads", type=int, default=8,
                           help="Multi Heads")
    transformer_arg.add_argument("--pf_layers", type=int, default=512,
                           help="Inner Embedding Dimension in Feedforward Layer")                       

    # Train
    train_arg = parser.add_argument_group('Training')
    train_arg.add_argument('--epochs', type=int, default=1000,
                           help='Number of epochs for model training')
    train_arg.add_argument('--nbatch', type=int, default=128,
                           help='Size of batch')
    train_arg.add_argument('--lr', type=float, default=1e-3,
                           help='Learning rate')
    train_arg.add_argument('--step_size', type=int, default=10,
                           help='Period of learning rate decay')
    train_arg.add_argument('--gamma', type=float, default=0.5,
                           help='Multiplicative factor of learning rate decay')
    train_arg.add_argument('--n_jobs', type=int, default=1,
                           help='Number of threads')
    train_arg.add_argument('--n_workers', type=int, default=1,
                           help='Number of workers for DataLoaders')

    return parser


def get_config():
    parser = get_parser()
    return parser.parse_known_args()[0]


