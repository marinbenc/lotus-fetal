import time
import torch, torchvision
import wandb
import numpy as np
from tqdm.auto import tqdm
from cut.cut_trainer import CUTTrainer
import helpers
from models.us_rendering_model import UltrasoundRendering

import matplotlib.pyplot as plt


class Generator:
    def __init__(self, hparams, opt_cut, plotter):
        self.hparams = hparams
        self.opt_cut = opt_cut
        self.plotter = plotter

        ModuleClass = helpers.load_module_class(hparams.module)
        InnerModelClass = helpers.load_model_class(hparams.inner_model)
        self.inner_model = InnerModelClass(params=hparams)
        
        if not hparams.outer_model_monai: 
            OuterModelClass = helpers.load_model_class(hparams.outer_model)
            outer_model = OuterModelClass(hparams=hparams)
            self.module = ModuleClass(params=hparams, inner_model=self.inner_model, outer_model=outer_model)
        else:
            self.module = ModuleClass(params=hparams, inner_model=self.inner_model)

        self.real_us_train_loader, dataset_real_us = helpers.load_real_us_training_data(opt_cut)
        self.cut_trainer = CUTTrainer(opt_cut, dataset_real_us, self.real_us_train_loader)
        self.USRendereDefParams = UltrasoundRendering(params=hparams, default_param=True).to(hparams.device)

    def generate(self, val_loader_ct_labelmaps):

        self.module.eval()
        self.module.outer_model.eval()
        self.inner_model.eval()
        self.cut_trainer.cut_model.eval()
        self.USRendereDefParams.eval()
        def_renderer_plot_figs = []
        print(f"--------------- VALIDATION SEG NET ------------")
        valid_losses = []
        with torch.no_grad():
            for nr, val_batch_data_ct in tqdm(enumerate(val_loader_ct_labelmaps), total=len(val_loader_ct_labelmaps), ncols= 100):
                
                val_input, val_label, filename = self.module.get_data(val_batch_data_ct)  

                # val_input_copy =  val_input.clone().detach()    
                us_rendr_val = self.module.rendering_forward(val_input)

                plt.imshow(val_input[0, 0].cpu().numpy(), cmap='gray')
                plt.title("Input Image")
                plt.show()
                plt.imshow(us_rendr_val[0, 0].cpu().numpy(), cmap='gray')
                plt.title("Rendered Image")
                plt.show() 


                print("Rendered us shape: ", us_rendr_val.shape)


                # val_loss_step, rendered_seg_pred = self.module.seg_forward(us_rendr_val, val_label)

                # valid_losses.append(val_loss_step.item())

                # if self.hparams.log_default_renderer and nr < self.hparams.nr_imgs_to_plot:
                #     us_rendr_def = self.USRendereDefParams(val_input_copy.squeeze()) 
                #     if not self.hparams.use_idtB: idt_B_val = us_rendr_val

                #     plot_fig = self.plotter.plot_stopp_crit(caption="labelmap|default_renderer|learnedUS|seg_input|seg_pred|gt",
                #                                 imgs=[val_input, us_rendr_def, us_rendr_val, rendered_seg_pred, val_label], 
                #                                 img_text='', epoch=epoch, plot_single=False)
                #     def_renderer_plot_figs.append(plot_fig)

                # elif not self.hparams.log_default_renderer:
                #     dict = self.module.plot_val_results(val_input, val_loss_step, filename, val_label, rendered_seg_pred, us_rendr_val, epoch)
                #     self.plotter.validation_batch_end(dict)

            # if len(def_renderer_plot_figs) > 0: 
            #     if self.hparams.logging:
            #         self.plotter.log_image(torchvision.utils.make_grid(def_renderer_plot_figs), "default_renderer|labelmap|defaultUS|learnedUS|seg_pred|gt")

        print(f"--------------- END SEG NETWORK VALIDATION ------------")
    
            

                    