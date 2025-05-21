import random
import wandb
import torch
from tqdm.auto import trange
import configargparse
from utils.plotter import Plotter
from utils.configargparse_arguments import build_configargparser
from utils.utils import argparse_summary
from cut.lotus_options import LOTUSOptions
import helpers
import generator

MANUAL_SEED = False


if __name__ == "__main__":

    if MANUAL_SEED: torch.manual_seed(2023)
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser.add('-c', is_config_file=True, help='config file path')
    parser, hparams = build_configargparser(parser)
    hparams.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    opt_cut = LOTUSOptions().parse()   # get training options
    opt_cut.dataroot = hparams.data_dir_real_us_cut_training
    opt_cut.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'torch.cuda.is_available(): {torch.cuda.is_available()}')

    # no horizontal flip for aorta_only segmentation task
    if hparams.aorta_only: opt_cut.no_flip = True

    if hparams.debug:
        hparams.exp_name = 'DEBUG'
    else:
        hparams.exp_name += str(hparams.pred_label) + '_' + str(opt_cut.lr) + '_' + str(hparams.inner_model_learning_rate) + '_' + str(hparams.outer_model_learning_rate)
    
    hparams.exp_name = str(random.randint(0, 1000)) + "_" + hparams.exp_name

    if hparams.logging: wandb.init(name=hparams.exp_name, project=hparams.project_name) #, silent=True)
    plotter = Plotter()

    argparse_summary(hparams, parser)
    # ---------------------
    # LOAD DATA 
    # ---------------------
    train_loader_ct_labelmaps, train_dataset_ct_labelmaps, val_dataset_ct_labelmaps, val_loader_ct_labelmaps = helpers.load_ct_labelmaps_training_data(hparams)
    _, real_us_stopp_crit_dataloader = helpers.load_real_us_gt_test_data(hparams)

    early_stopping = helpers.create_early_stopping(hparams, hparams.exp_name, 'valid')

    avg_train_losses, avg_valid_losses = ([] for i in range(2))

    # --------------------
    # RUN TRAINING
    # ---------------------

    generator = generator.Generator(hparams, opt_cut, plotter)
    generator.cut_trainer.cut_model.set_requires_grad(generator.USRendereDefParams, False)

    seg_network_ckpt = 'checkpoints/best_checkpoint_seg_renderer_valid_loss_335_exp_name13_5e-06_0.0001_0.0001_epoch=1.pt'     #replace with your ckpt after training
    generator.module.load_state_dict(torch.load(seg_network_ckpt))
    
    cut_network_ckpt = 'checkpoints/best_checkpoint_CUT_val_loss_335_exp_name13_5e-06_0.0001_0.0001_epoch=1.pt'     #replace with your ckpt after training
    checkpoint = torch.load(cut_network_ckpt)
    # Create a new dictionary with keys without the "module." prefix
    new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    generator.cut_trainer.cut_model.netG.load_state_dict(new_state_dict)

    generator.generate(val_loader_ct_labelmaps)