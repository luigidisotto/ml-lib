import sys
import logging
import argparse
from timeit import default_timer as timer

import numpy as np

from nn import MLP, HyperOptimizer, MeanSquaredError, Linear, Sigmoid
from utils.plot import scatter_plot, plot_accuracy, plot_perf
from monk import retrieveSamplesFromMonk

logger = logging.getLogger(__name__)


def main():

    # Learning the XOR
    # (obs.: sensitive to standard dev. we use to init. weights)
    # X = np.array([[.1, .1], [.1, .9], [.9, .1], [.9, .9]], dtype=float)
    # Y = np.array([[.1], [.9], [.9], [.1]], dtype=float)
    # xor = MLP(learning_rate=1.0, layers=[2, 2, 1], loss=MeanSquaredError, sigma=0.05)
    # perf_xor = xor.fit(X, Y)
    # plot_perf(perf_xor, 'perf_xor')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bench_name",
        type=str,
        help="To choose the task. Available: monks-1, monks-2, monks-3."
    )
    parser.add_argument(
        "--do_grid_search",
        action="store_true",
        help="Wheter to do hyper-params search via grid-search."
    )
    parser.add_argument(
        "--lr",
        default=1e-1,
        type=float,
        help="The learning rate. Default: 1e-1.",
    )
    parser.add_argument(
        "--momentum",
        default=0.0,
        type=float,
        help="The momentum. Default: 0.0.",
    )
    parser.add_argument(
        "--reg",
        default=0.0,
        type=float,
        help="The regularization term. Default: 0.0.",
    )
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="The maximum number of epochs.",
    )
    parser.add_argument(
        "--sigma",
        default=1.0,
        type=float,
        help="To init weights with normal distribution N(mu, sigma), and scale sigma.",
    )
    parser.add_argument(
        "--mu",
        default=0.0,
        type=float,
        help="To init weights with normal distribution N(mu, sigma), and center mu.",
    )
    parser.add_argument(
        '--layers',
        nargs='+',
        help='The layers\' inputs. E.g. n_input n_hid_i n_hid_k n_class, where n_input is the size of the input to the MLP,\
		n_hid_j is the size of hidden layers, and n_class is the size of last layer for regression/classification tasks.'
    )
    parser.add_argument(
        "--activation",
        default='Linear',
        type=str,
        help="The activation function. Available: Linear, Sigmoid."
    )
    parser.add_argument(
        "--activation_last_layer",
        default='Sigmoid',
        type=str,
        help="The activation function to be applied after the classification or regression layer. Available: Linear, Sigmoid."
    )
    parser.add_argument(
        "--task",
        type=str,
        help="The type of task to be solved. Available: classification, regression."
    )
    parser.add_argument(
        "--verbose",
        default='no',
        type=str,
        help="If output in verbose mode."
    )
    parser.add_argument(
        "--debug_interval",
        default=100,
        type=int,
        help="To output training info every debug_interval epochs."
    )
    parser.add_argument(
        "--do_early_stopping",
        action="store_true",
        help="Wheter to do early stopping."
    )

    args = parser.parse_args()

    X_train, Y_train = retrieveSamplesFromMonk(
        f"monks-data/{args.bench_name}.train")
    X_test, Y_test = retrieveSamplesFromMonk(
        f"monks-data/{args.bench_name}.test")

    logger.info(f"Training set len = {len(X_train)}")
    logger.info(f"Test set len = {len(X_test)}")

    logger.warning(
        "do_grid_search: %s",
        bool(args.do_grid_search)
    )

    if args.do_grid_search:
        n_input, n_output = 17, 1
        hyperspace = {
            'learning_rate': [i/10 for i in range(1, 10)],
            'momentum': [i/10 for i in range(1, 10)],
            'lambda': [0.0],  # [10**(-i) for i in range(1,6)],
            'topology': [[n_input, 2**j, n_output] for j in range(1, 6)]
        }
        logger.info('Starting hyper-parameter grid-search.')
        start = timer()
        monk = HyperOptimizer(
            hyperspace,
            X_train,
            Y_train,
            loss=MeanSquaredError,
            p=.70,
            k=5,
            scoring=['valid_err', 'test_err', 'accuracy'],
            task=args.task,
            X_ts=X_test,
            Y_ts=Y_test
        )
        monk.hoptimize()
        end = timer()
        logger.info(
            f'Grid-search for task {args.bench_name} has taken = {np.ceil((end-start)*1000)} ms')

    monk = MLP(
        learning_rate=args.lr,
        momentum=args.momentum,
        lamda=args.reg,
        max_epochs=args.max_epochs,
        layers=[int(l) for l in args.layers],
        task=args.task,
        verbose=args.verbose,
        activation=Linear if args.activation == 'Linear' else Sigmoid,
        activation_out=Linear if args.activation_last_layer == 'Linear' else Sigmoid,
        do_early_stopping=args.do_early_stopping,
        debug_interval=args.debug_interval
    )

    logger.info(f'Starting training for task {args.bench_name}.')
    start = timer()
    perf_monk = monk.fit(X_train, Y_train)
    end = timer()
    logger.info(
        f'Training for task {args.bench_name} has taken = {np.ceil((end-start)*1000)} ms')

    test_err = monk.score(X_test, Y_test)
    logger.info(f'Test error = {test_err}')

    plot_perf(perf_monk, f'perf_{args.bench_name}')
    plot_accuracy(perf_monk, f'acc_{args.bench_name}')


if __name__ == "__main__":
    main()
