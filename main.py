import argparse
import logging, os
import sys
import pdb
from pathlib import Path
import torch
from core.utils import Logger
from torch.utils.data import DataLoader
from torch import optim, nn
from core import build_model, build_dataset, train
from core.train import train_epoch
from core.validation import val_epoch
from core.test import test
from core.build_model import MODELS


def run(args, logger):
    # Build model
    epoch_start = 0
    load_optimizer = 0
    checkpoint = None

    if args.check_resume and args.do_train:
        checkpoint_path = os.path.join(args.output_dir, 'save_last.pth')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            load_optimizer = True
            epoch_start = checkpoint['epoch']

            logger.info(f"Found previous checkpoint, loading and starting at {epoch_start}")

            if epoch_start == args.epochs:
                logger.info(f"Model completed training. If this is not expected, remove `check_resume` flag."
                            f" Skipping training...")
                return
            args.pretrain_pth = checkpoint_path
            args.pretrain_dataset = args.train_dataset

    logger.info(f"Starting building {args.model_type}...")
    model, parameters = build_model.fetch_model(logger, args.model_type, train_dataset=args.train_dataset,
                                                pretrain_dataset=args.pretrain_dataset,
                                                pretrained_pth=args.pretrain_pth)
    model.cuda()

    logger.info(f"Completed building {args.model_type}")
    test_dataset = build_dataset.ActionRecognitionUniformFrames(args, 'test')
    # Test on different perturbation dataset
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.test_batch_size,
                                 shuffle=False,
                                 num_workers=args.num_workers)

    if args.nesterov:
        dampening = 0
    else:
        dampening = args.dampening

    if args.do_train:
        train_dataset = build_dataset.ActionRecognitionUniformFrames(args, 'train')
        val_dataset = build_dataset.ActionRecognitionUniformFrames(args, 'val')

        # Perturbation being trained on and validation dataset for that
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.train_batch_size,
                                      shuffle=True,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=args.test_batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)

        train_logger = Logger(
            os.path.join(args.output_dir, 'train.log'),
            ['epoch', 'loss', 'acc', 'lr'])
        train_batch_logger = Logger(
            os.path.join(args.output_dir, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'acc', 'lr'])
        val_logger = Logger(
            os.path.join(args.output_dir, 'val.log'), ['epoch', 'loss', 'acc'])

        logger.info("Initializing optimizer and learning schedule.")

        optimizer = optim.SGD(
            parameters,
            lr=args.learning_rate,
            momentum=args.momentum,
            dampening=dampening,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov)

        if load_optimizer:
            logger.info(f"Loading previous optimizer at {checkpoint_path}")
            optimizer.load_state_dict(checkpoint['optimizer'])

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=args.lr_patience)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epoch_start, args.epochs):
            train_epoch(epoch, train_dataloader, model, criterion, optimizer, args, train_logger, train_batch_logger)
            val_loss = val_epoch(epoch, val_dataloader, model, criterion, args,val_logger)
            scheduler.step(val_loss)

    logger.info(f"Running testing on {args.test_perturbation} on {args.test_severity}.")
    model.eval()
    test(logger, test_dataloader, model, args, test_dataset.classnames)



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs training on perturbed datasets for selected models.')

    # Model specific arguments
    parser.add_argument('model_type', type=str,
                        help=f"The model type you would like to use. Options are {' ,'.join(MODELS)}")
    parser.add_argument('--pretrain_pth', default=None, type=str)

    # Run configurations
    parser.add_argument('--output_dir', default='output', type=str,
                        help='Where you would like to store the results.')
    parser.add_argument('--do_train', const=True, default=False, action="store_const")
    parser.add_argument('--no_softmax_in_test', const=True, default=False, action="store_const")
    parser.add_argument('--num_workers', default=4, type=int)

    # Dataset specific
    parser.add_argument('--pretrain_dataset', default=None, type=str,
                        help='What dataset the model was pre-trained on. This determines the linear prediction head on '
                             'initial build.')
    parser.add_argument('--train_dataset', default='ucf101', type=str,
                        help='What dataset to train a model on. This determines the change in the linear prediction '
                             'head for training.')
    parser.add_argument('--test_dataset', default='ucf101', type=str)
    parser.add_argument('--train_perturbation', default=None, type=str,
                        help='Which set of perturbations to use on the training data. Options can be found in '
                             '`video_perturb.py` but are typically `mixed`, `spatial`, `temporal`, or `pixmix`.')
    parser.add_argument('--train_severity', default=None, type=int, help='If choosing just one type of perturbation, '
                                                                         'you should choose a severity as well.')
    parser.add_argument('--test_perturbation', default=None, type=str)
    parser.add_argument('--test_severity', default=None, type=int)

    # Video extraction specific
    parser.add_argument('--root_dir', default='/media/mschiappa/Elements/UCF101/videos', type=str)
    # parser.add_argument('--num_frames', default=16, type=int)
    # parser.add_argument('--input_res', default=112, type=int)
    parser.add_argument('--sample_type', default='uniform', type=str)
    parser.add_argument('--fix_start', const=True, default=False, action="store_const")
    parser.add_argument('--train_path', default='/media/mschiappa/Elements/UCF101/trainlist01.txt', type=str,
                        help='training annotations for the dataset being trained on.')
    parser.add_argument('--test_path', default='/media/mschiappa/Elements/UCF101/vallist01.txt', type=str,
                        help='testing annotations for the dataset being trained on.')

    # Train configurations
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--checkpoint', default=10, type=int)
    parser.add_argument('--check_resume', const=True, default=False, action="store_const")
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=32, type=int)
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--nesterov', action='store_true', help='Nesterov momentum')
    parser.add_argument('--lr_patience', default=10, type=int,
                        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')

    args = parser.parse_args()

    if args.pretrain_pth is not None: # and args.do_train:
        output_dir = os.path.join(args.output_dir, args.model_type, f'ft_{args.train_dataset}',
                                  f"{args.train_perturbation}_{args.train_severity}")
    else:
        output_dir = os.path.join(args.output_dir, args.model_type, f'scratch_{args.train_dataset}',
                                  f"{args.train_perturbation}_{args.train_severity}")

    log_path = os.path.join(output_dir, 'log.txt')
    if not os.path.exists(output_dir):
        Path(output_dir).mkdir(parents=True)

    if os.path.isfile(log_path) and os.path.isfile(os.path.join(output_dir, 'save_50.pth')):
        print(f"Model already exists. Exiting...")
        sys.exit()

    if os.path.isfile(log_path) and not args.do_train:
        log_path = os.path.join(output_dir, f'test_{args.test_perturbation}_{args.test_severity}.txt')
        # if os.path.isfile(log_path):
        #     print(f"Test json file already exists. Exiting...")
        #     sys.exit()

    args.output_dir = output_dir

    # Create a Logger Object - Which listens to everything
    logger = logging.getLogger(os.path.basename(__file__))
    logger.setLevel(logging.DEBUG)

    # Register the Console as a handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Log format includes date and time
    formatter = logging.Formatter('%(asctime)s %(levelname)-5s %(message)s')
    ch.setFormatter(formatter)

    # If want to print output to screen
    logger.addHandler(ch)

    # Create a File Handler to listen to everything
    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.DEBUG)

    # Log format includes date and time
    fh.setFormatter(formatter)

    # Register it as a listener
    logger.addHandler(fh)

    # Print arguments
    logger.info(f"Storing log path in {log_path}")
    logger.info("Model configurations")
    logger.info('-------------------------')
    for arg in vars(args):
        logger.info(f"{arg}: {getattr(args, arg)}")

    run(args, logger)
