import argparse

import torch
from IPython.core.debugger import set_trace
from torch import nn
from torch.nn import functional as F
from data import data_helper
# from IPython.core.debugger import set_trace
from data.data_helper import available_datasets
from models import model_factory
from optimizer.optimizer_helper import get_optim_and_scheduler
from utils.Logger import Logger
import numpy as np
import pdb

def get_args():
    parser = argparse.ArgumentParser(description="Script to launch jigsaw training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", choices=available_datasets, help="Source", nargs='+')
    parser.add_argument("--target", choices=available_datasets, help="Target")
    parser.add_argument("--batch_size", "-b", type=int, default=64, help="Batch size")
    parser.add_argument("--image_size", type=int, default=225, help="Image size")
    # data aug stuff
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float, help="Chance of randomly greyscaling a tile")
    #
    parser.add_argument("--limit_source", default=None, type=int, help="If set, it will limit the number of training samples")
    parser.add_argument("--limit_target", default=None, type=int, help="If set, it will limit the number of testing samples")

    parser.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_classes", "-c", type=int, default=31, help="Number of classes")
    parser.add_argument("--jigsaw_n_classes", "-jc", type=int, default=31, help="Number of classes for the jigsaw task")
    parser.add_argument("--network", choices=model_factory.nets_map.keys(), help="Which network to use", default="caffenet")
    parser.add_argument("--jig_weight", type=float, default=0.1, help="Weight for the jigsaw puzzle")
    parser.add_argument("--rex_weight_class", type=float, default=0.1, help="Weight for rex on classsification risk")
    parser.add_argument("--irm_weight_class", type=float, default=0.1, help="Weight for rex on classsification risk")
    parser.add_argument("--rex_weight_jigsaw", type=float, default=0.1, help="Weight for rex on jigsaw risk")
    parser.add_argument("--irm_weight_jigsaw", type=float, default=0.1, help="Weight for rex on jigsaw risk")
    parser.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
    parser.add_argument("--tf_logger", type=bool, default=True, help="If true will save tensorboard compatible logs")
    parser.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
    parser.add_argument("--folder_name", default=None, help="Used by the logger to save logs")
    parser.add_argument("--bias_whole_image", default=None, type=float, help="If set, will bias the training procedure to show more often the whole image")
    parser.add_argument("--TTA", action='store_true', help="Activate test time data augmentation")
    parser.add_argument("--classify_only_sane", action='store_true',
                        help="If true, the network will only try to classify the non scrambled images")
    parser.add_argument("--train_all", action='store_true', help="If true, all network weights will be trained")
    parser.add_argument("--suffix", default="", help="Suffix for the logger")
    parser.add_argument("--nesterov", action='store_true', help="Use nesterov")
    parser.add_argument("--adam", action='store_true', help="Use nesterov")
    parser.add_argument("--prefix", type=str)
    
    return parser.parse_args()


# def compute_losses(net_output, jig_l, class_l):
#     return F.cross_entropy(net_output[0], jig_l), F.cross_entropy(net_output[1], class_l)

def compute_rex_penalty(r1,r2,r3):
    mu = (r1 + r2 + r3)/3
    return ((r1-mu)**2 + (r2-mu)**2 + (r3-mu)**2)/3

def mean_nll(logits, y, criterion):
    return criterion(logits, y)

def compute_irm_penalty(logits, y, criterion):
    scale = torch.ones_like(logits).requires_grad_()
    loss = mean_nll(logits * scale, y, criterion)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2) 

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        model = model_factory.get_network(args.network)(jigsaw_classes=args.jigsaw_n_classes + 1, classes=args.n_classes)
        self.model = model.to(device)
        # print(self.model)
        self.source_loader, self.val_loader = data_helper.get_train_dataloader(args, patches=model.is_patch_based())
        self.target_loader = data_helper.get_val_dataloader(args, patches=model.is_patch_based())
        self.test_loaders = {"val": self.val_loader, "test": self.target_loader}
        self.len_dataloader = len(self.source_loader)
        print("Dataset size: train %d, val %d, test %d" % (len(self.source_loader.dataset), len(self.val_loader.dataset), len(self.target_loader.dataset)))
        self.optimizer, self.scheduler = get_optim_and_scheduler(model, args.epochs, args.learning_rate, args.train_all, nesterov=args.nesterov, adam=args.adam)
        self.jig_weight = args.jig_weight
        self.rex_weight_class = args.rex_weight_class
        self.irm_weight_class = args.irm_weight_class
        self.rex_weight_jigsaw = args.rex_weight_jigsaw
        self.irm_weight_jigsaw = args.irm_weight_jigsaw
        self.only_non_scrambled = args.classify_only_sane
        self.n_classes = args.n_classes
        if args.target in args.source:
            self.target_id = args.source.index(args.target)
            print("Target in source: %d" % self.target_id)
            print(args.source)
        else:
            self.target_id = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for it, ((data, jig_l, class_l), d_idx) in enumerate(self.source_loader):
            data, jig_l, class_l, d_idx = data.to(self.device), jig_l.to(self.device), class_l.to(self.device), d_idx.to(self.device)
            self.optimizer.zero_grad()

            jigsaw_logit, class_logit = self.model(data)  # , lambda_val=lambda_val)
            j1 = criterion(jigsaw_logit[d_idx == 0], jig_l[d_idx == 0])
            j1_irm = compute_irm_penalty(jigsaw_logit[d_idx == 0], jig_l[d_idx == 0], criterion)
            j2 = criterion(jigsaw_logit[d_idx == 1], jig_l[d_idx == 1])
            j2_irm = compute_irm_penalty(jigsaw_logit[d_idx == 1], jig_l[d_idx == 1], criterion)
            j3 = criterion(jigsaw_logit[d_idx == 2], jig_l[d_idx == 2])
            j3_irm = compute_irm_penalty(jigsaw_logit[d_idx == 2], jig_l[d_idx == 2], criterion)
            rex_jigsaw = compute_rex_penalty(j1,j2,j3)
            jigsaw_loss = j1+j2+j3
            irm_jigsaw = (j1_irm+j2_irm+j3_irm)/3
            if self.only_non_scrambled:
                if self.target_id is not None:
                    idx = (jig_l == 0) & (d_idx != self.target_id)
                    class_loss = criterion(class_logit[idx], class_l[idx])
                    rex_class = torch.Tensor([0.]).cuda()
                    irm_class = torch.Tensor([0.]).cuda()
                else:
                    class_loss_1 = criterion(class_logit[(jig_l == 0) & (d_idx == 0)], class_l[(jig_l == 0) & (d_idx == 0)])
                    class_irm_1 = compute_irm_penalty(class_logit[(jig_l == 0) & (d_idx == 0)], class_l[(jig_l == 0) & (d_idx == 0)], criterion)
                    class_loss_2 = criterion(class_logit[(jig_l == 0) & (d_idx == 1)], class_l[(jig_l == 0) & (d_idx == 1)])
                    class_irm_2 = compute_irm_penalty(class_logit[(jig_l == 0) & (d_idx == 1)], class_l[(jig_l == 0) & (d_idx == 1)], criterion)
                    class_loss_3 = criterion(class_logit[(jig_l == 0) & (d_idx == 2)], class_l[(jig_l == 0) & (d_idx == 2)])
                    class_irm_3 = compute_irm_penalty(class_logit[(jig_l == 0) & (d_idx == 2)], class_l[(jig_l == 0) & (d_idx == 2)], criterion)
                    class_loss = class_loss_1 + class_loss_2 + class_loss_3
                    irm_class = (class_irm_1 + class_irm_2 + class_irm_3)/3
                    rex_class = compute_rex_penalty(class_loss_1, class_loss_2, class_loss_2) 

            elif self.target_id:
                class_loss = criterion(class_logit[d_idx != self.target_id], class_l[d_idx != self.target_id])
                rex_class = torch.Tensor([0.]).cuda()
                irm_class = torch.Tensor([0.]).cuda()
            else:
                class_loss_1 = criterion(class_logit[(d_idx == 0)], class_l[(d_idx == 0)])
                class_irm_1 = compute_irm_penalty(class_logit[(d_idx == 0)], class_l[(d_idx == 0)], criterion)
                class_loss_2 = criterion(class_logit[(d_idx == 1)], class_l[(d_idx == 1)])
                class_irm_2 = compute_irm_penalty(class_logit[(d_idx == 1)], class_l[(d_idx == 1)], criterion)
                class_loss_3 = criterion(class_logit[(d_idx == 2)], class_l[(d_idx == 2)])
                class_irm_3 = compute_irm_penalty(class_logit[(d_idx == 2)], class_l[(d_idx == 2)], criterion)
                class_loss = class_loss_1 + class_loss_2 + class_loss_3
                irm_class = (class_irm_1 + class_irm_2 + class_irm_3)/3
                rex_class = compute_rex_penalty(class_loss_1, class_loss_2, class_loss_2) 
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)


            rex_loss = self.rex_weight_class * rex_class + self.rex_weight_jigsaw * self.jig_weight * rex_jigsaw
            irm_loss = self.irm_weight_class * irm_class + self.irm_weight_jigsaw * self.jig_weight * irm_jigsaw
            if self.rex_weight_class == 0. and self.rex_weight_jigsaw == 0. and self.irm_weight_jigsaw == 0. and self.irm_weight_class == 0.:
                loss = class_loss + jigsaw_loss * self.jig_weight
            elif self.irm_weight_jigsaw == 0. and self.irm_weight_class == 0.:
                loss = class_loss + jigsaw_loss * self.jig_weight + rex_loss 
            elif self.rex_weight_class == 0. and self.rex_weight_jigsaw == 0.:
                loss = class_loss + jigsaw_loss * self.jig_weight + irm_loss 

            loss.backward()
            self.optimizer.step()

            self.logger.log(it, len(self.source_loader),
                            {"jigsaw": jigsaw_loss.item(), "class": class_loss.item(), "rex loss class": rex_class.item(), "rex loss jigsaw": rex_jigsaw.item(), "rext total": rex_loss.item(), "irm loss class": irm_class.item(), "irm loss jigsaw": irm_jigsaw.item(), "irm total": irm_loss.item()},
                            # ,"lambda": lambda_val},
                            {"jigsaw": torch.sum(jig_pred == jig_l.data).item(),
                             "class": torch.sum(cls_pred == class_l.data).item(),
                             # "domain": torch.sum(domain_pred == d_idx.data).item()
                             },
                            data.shape[0])
            del loss, class_loss, jigsaw_loss, rex_loss, jigsaw_logit, class_logit

        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc

    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct

    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.n_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct

    def do_training(self):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            self.logger.new_epoch(self.scheduler.get_lr())
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()
        #print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        name = self.args.prefix+"_"+str(self.args.source[0])+str(self.args.source[1])+str(self.args.source[2])+"_"+str(self.args.target)+"_eps%d_bs%d_lr%g_class%d_jigClass%d_rexWeightClass%g_rexWeightJig%g_irmWeightClass%g_irmWeightJig%g_jigWeight%g" % (self.args.epochs, self.args.batch_size, self.args.learning_rate, self.args.n_classes, self.args.jigsaw_n_classes, self.args.rex_weight_class, self.args.rex_weight_jigsaw, self.args.irm_weight_class, self.args.irm_weight_jigsaw, self.args.jig_weight)
        with open('./result_summary_txt/'+name+'.txt', 'a+') as f:
            f.write('best validation accuracy: '+str(val_res.max())+' test acc at best val acc: '+str(test_res[idx_best])+' max test: '+str(test_res.max()))
        self.logger.save_best(test_res[idx_best], test_res.max())
        return self.logger, self.model


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(args, device)
    trainer.do_training()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
