import pdb
import time
import importlib.util
import os
import logging
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import sys
sys.path.append('./')
# from utils.model_utils import optim_list, count_parameters
# from utils.misc_utils import create_directory, get_by_dotted_path, add_record, get_records, log_record_dict
# from utils.plot_utils import create_curve_plots
import numpy as np
import os

disp_avlbl = True
from os import environ

if 'DISPLAY' not in environ:
    disp_avlbl = False
    import matplotlib

    matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plotCurve(values, timesteps=None, errs=None, savefile=None, fontsize=None, **kwargs):
    ''' Plots values vs iterations

    Args:
        values: Values to plot (y-axis)
        timesteps: X-axis labels
        errs: Y-axis errors to plot errorbars
        savefile: If not None, plot is saved to this file
        fontsize: Font size for plot labels
    '''
    # Match array lengths
    n_steps = len(values)
    if timesteps is not None:
        assert len(values) == len(timesteps)

    # Set font size
    if fontsize is not None:
        default_fontsize = plt.rcParams.get('font.size')
        plt.rcParams.update({'font.size': fontsize})

    # Plot curves
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    if errs is None:
        ax.plot(values, marker='o')
    else:
        ax.errorbar(range(n_steps), values, yerr=errs, fmt='o-')

    # Labels
    xticks = timesteps if timesteps is not None else range(1, n_steps + 1)
    plt.xticks(range(n_steps), xticks)
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else 'Step')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else 'Value')

    # Save/display figure
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    # Reset font size
    if fontsize is not None:
        plt.rcParams.update({'font.size': default_fontsize})
    plt.close()


def plotCurves(curves, timesteps=None, errs=None, legend_labels=None, savefile=None, fontsize=None, **kwargs):
    ''' Plots multiple curves sharing their x-axis

    Args:
        curves: List of curves to plot (y-axis)
        timesteps: X-axis labels
        errs: List of Y-axis errs for each curve in curves
        legend_labels: Legends for each curve
        savefile: If not None, plot is saved to this file
        fontsize: Font size for plot labels
    '''
    # Match lengths
    num_curves = len(curves)
    if legend_labels is not None:
        assert len(legend_labels) == num_curves
    if errs is not None:
        assert num_curves == len(errs)

    n_steps = len(curves[0])
    for curve in curves:
        assert n_steps == len(curve)
    if timesteps is not None:
        assert n_steps == len(timesteps)
    if errs is not None:
        for err in errs:
            assert n_steps == len(err)

            # Set font size
    if fontsize is not None:
        default_fontsize = plt.rcParams.get('font.size')
        plt.rcParams.update({'font.size': fontsize})

    # Plot curves
    fig = plt.figure()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    handles = []

    if 'yscale' in kwargs:
        # ax.set_yscale("log", nonposy='clip')
        ax.set_yscale(kwargs['yscale'], nonposy='clip')

    for i in range(num_curves):
        label = legend_labels[i] if legend_labels is not None else None
        if errs is None:
            h, = ax.plot(curves[i], marker='o', label=label)
        else:
            h, _, _ = ax.errorbar(range(n_steps), curves[i], yerr=errs[i], fmt='o-', label=label)
        handles.append(h)

    # Labels
    xticks = timesteps if timesteps is not None else range(1, n_steps + 1)
    plt.xticks(range(n_steps), xticks)
    plt.xlabel(kwargs['xlabel'] if 'xlabel' in kwargs else 'Step')
    plt.ylabel(kwargs['ylabel'] if 'ylabel' in kwargs else 'Value')
    if legend_labels is not None:
        # plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # plt.legend(handles=handles)
        # plt.legend()
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

    # Save/display figure
    if savefile is not None:
        plt.savefig(savefile, bbox_inches='tight')

    # Reset font size
    if fontsize is not None:
        plt.rcParams.update({'font.size': default_fontsize})
    plt.close()


def curve_plot(values_dict):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    for key, values in values_dict.items():
        if values == []:
            continue
        ax.plot(values, label=key)
        ax.set_xlabel('epochs')
        ax.set_title('plot')
        ax.legend()
    return fig


def create_curve_plots(name, plot_dict, log_dir):
    fig = curve_plot(plot_dict)
    fig.suptitle(name)
    fig.savefig(os.path.join(log_dir, name + '_curve.png'), bbox_inches='tight', pad_inches=0)
    plt.close(fig)

############################ Base model class ###########################

import os
import errno


#### Files and Directories ####

def delete_files(folder, recursive=False):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif recursive and os.path.isdir(file_path):
                delete_files(file_path, recursive)
                os.unlink(file_path)
        except Exception as e:
            print(e)


def create_directory(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def get_files(dirpath):
    return [name for name in os.listdir(dirpath) if os.path.isfile(os.path.join(dirpath, name))]


def get_dirs(dirpath):
    return [name for name in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, name))]


#### Logging Dictionary Tools ####

def get_by_dotted_path(d, path, default=[]):
    """ Get an entry from nested dictionaries using a dotted path.

    Args:
        d: Dictionary
        path: Entry to extract

    Example:
    >>> get_by_dotted_path({'foo': {'a': 12}}, 'foo.a')
    12
    """
    if not path:
        return d
    split_path = path.split('.')
    current_option = d
    for p in split_path:
        if p not in current_option:
            return default
        current_option = current_option[p]
    return current_option


def add_record(key, value, global_logs):
    if 'logs' not in global_logs['info']:
        global_logs['info']['logs'] = {}
    logs = global_logs['info']['logs']
    split_path = key.split('.')
    current = logs
    for p in split_path[:-1]:
        if p not in current:
            current[p] = {}
        current = current[p]

    final_key = split_path[-1]
    if final_key not in current:
        current[final_key] = []
    entries = current[final_key]
    entries.append(value)


def get_records(key, global_logs):
    logs = global_logs['info'].get('logs', {})
    return get_by_dotted_path(logs, key)


def log_record_dict(usage, log_dict, global_logs):
    for log_key, value in log_dict.items():
        add_record('{}.{}'.format(usage, log_key), value, global_logs)


#### Controlling verbosity ####

def vprint(verbose, *args, **kwargs):
    ''' Prints only if verbose is True.
    '''
    if verbose:
        print(*args, **kwargs)


def vcall(verbose, fn, *args, **kwargs):
    ''' Calls function fn only if verbose is True.
    '''
    if verbose:
        fn(*args, **kwargs)

optim_list = {
    'sgd': torch.optim.SGD,
    'adagrad': torch.optim.Adagrad,
    'adadelta': torch.optim.Adadelta,
    'adam': torch.optim.Adam,
    'adamax': torch.optim.Adamax,
    'rmsprop': torch.optim.RMSprop,
}
def count_parameters(net):
    """ Returns total number of trainable parameters in net """
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class BaseModel(object):

    def __init__(self, Net, device, global_records, config):
        # Initializations
        self.device = device
        self.global_records = global_records
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize network
        self.net = Net(**self.config['net'])

        # Then load its params if available
        if self.config['net'].get('saved_params_path', None) is not None:
            self.load_net(self.config['net']['saved_params_path'])

        # Initialize optimizer
        self.setup_optimizer()

        # Initialize learning rate scheduler
        self.setup_lr_scheduler()

        # Transfer network to device
        self.net.to(self.device)
        self.logger.info(self.net)
        self.logger.info("Number of parameters: %d" % (count_parameters(self.net)))

        # Losses for all models (more can be defined in derived models if needed)
        self.mse_loss_fn = nn.MSELoss(reduction='none')
        self.mae_loss_fn = nn.L1Loss(reduction='none')

        # Initialize epoch number
        self.epoch = 0

    # Abstract method to implement
    def run_batch(self, batch, mode='train', store_losses=True, store_debug=False):
        raise NotImplementedError()

    def setup_optimizer(self):
        self.optimizer = None
        if 'optimizer' in self.config:
            optim = optim_list[self.config['optimizer']['name']]
            self.optimizer = optim(self.net.parameters(), **self.config['optimizer']['params'])

    def setup_lr_scheduler(self):
        self.scheduler = None
        if self.config['optimizer'].get('scheduler'):
            self.scheduler = lr_scheduler.StepLR(self.optimizer, **self.config['optimizer']['scheduler'])
            self.logger.debug('Using LR scheduler: '+ str(self.config['optimizer']['scheduler']))

    def step_lr_scheduler(self):
        if self.scheduler:
            self.scheduler.step()
            self.logger.debug("Learning rate: %s" % (','.join([str(lr) for lr in self.scheduler.get_lr()])))

    def fit(self, tr_loader, val_loader, *args, **kwargs):
        # Initialize params
        if 'max_epoch' in kwargs:
            max_epoch = kwargs['max_epoch']
        else:
            assert 'max_epoch' in self.config['train']['stop_crit'], "max_epoch not specified in config['train']['stop_crit']"
            max_epoch = self.config['train']['stop_crit']['max_epoch']

        if 'min_epoch' in kwargs:
        	min_epoch = kwargs['min_epoch']
        elif 'min_epoch' in self.config['train']['stop_crit']:
        	min_epoch = self.config['train']['stop_crit']['min_epoch']
        else:
        	min_epoch = max_epoch // 2

        if 'max_patience' in kwargs:
            max_patience = kwargs['max_patience']
        elif 'max_patience' in self.config['train']['stop_crit']:
            max_patience = self.config['train']['stop_crit']['max_patience']
        else:
            max_patience = None
        if max_patience is not None:
            assert max_patience > 0, "max_patience should be positive"
            self.logger.debug('Early stopping enabled with max_patience = {}'.format(max_patience))
        else:
            self.logger.debug('Early stopping disabled since max_patience not specified.')

        # Train epochs
        best_valid_loss = np.inf
        best_valid_epoch = 0
        early_break = False
        for epoch in range(max_epoch):
            self.logger.info('\n' + 40 * '%' + '  EPOCH {}  '.format(epoch) + 40 * '%')
            self.epoch = epoch

            # Perform LR scheduler step
            self.step_lr_scheduler()

            # Run train epoch
            t = time.time()
            epoch_records = self.run_epoch(tr_loader, 'train', epoch, *args, **kwargs)

            # Log and print train epoch records
            log_record_dict('train', epoch_records, self.global_records)
            self.print_record_dict(epoch_records, 'Train', time.time() - t)
            self.global_records['result'].update({
                'final_train_loss': epoch_records['loss'],
                'final_train_epoch': epoch
            })

            if val_loader is not None:
                # Run valid epoch
                t = time.time()
                epoch_records = self.run_epoch(val_loader, 'eval', epoch, *args, **kwargs)

                # Log and print valid epoch records
                log_record_dict('valid', epoch_records, self.global_records)
                self.print_record_dict(epoch_records, 'Valid', time.time() - t)
                self.global_records['result'].update({
                    'final_valid_loss': epoch_records['loss'],
                    'final_valid_epoch': epoch
                })

                # Check for early-stopping
                if epoch_records['loss'] < best_valid_loss:
                    best_valid_loss = epoch_records['loss']
                    best_valid_epoch = epoch
                    self.global_records['result'].update({
                        'best_valid_loss': best_valid_loss,
                        'best_valid_epoch': best_valid_epoch
                    })
                    self.logger.info('    Best validation loss improved to {:.8f}'.format(best_valid_loss))
                    self.save_net(os.path.join(self.config['outdir'], 'best_valid_params.ptp'))

                if (epoch > min_epoch) and (max_patience is not None) and (best_valid_loss < np.min(get_records('valid.loss', self.global_records)[-max_patience:])):
                    early_break = True

            # Produce plots
            plots = self._plot_helper(epoch_records) # Needs epoch_records for names of logged losses
            if plots is not None:
                for k, v in plots.items():
                    create_curve_plots(k, v, self.config['outdir'])

            # Save net
            self.save_net(os.path.join(self.config['outdir'], 'final_params.ptp'))

            # Save results
            pickle.dump(self.global_records, file=open(os.path.join(self.config['outdir'], self.config['record_file']), 'wb'))

            # Early-stopping
            if early_break:
                self.logger.warning('Early Stopping because validation loss did not improve for {} epochs'.format(max_patience))
                break

    def run_epoch(self, data_loader, mode, epoch, *args, **kwargs):
        epoch_losses = {}
        num_batches = len(data_loader)
        
        # Iterate over batches
        for batch_idx, batch in enumerate(data_loader):
            # Eval options
            store_losses = self.config['eval'].get('store_losses', True)
            store_debug = self.config['eval'].get('store_debug', False)
            
            # Run the batch
            batch_info = self.run_batch(batch, mode, store_losses, store_debug)
            batch_losses = batch_info['losses'] if store_losses else None
            batch_debug = batch_info['debug'] if store_debug else None
            
            # Log stuff
            log = self.config['log_interval']
            if batch_idx % log == 0:
                loss_vals = ''
                if batch_losses is not None:
                    for loss in batch_losses:
                        mean_loss = batch_losses[loss]['val'] / batch_losses[loss]['numel'] if batch_losses[loss]['numel'] != 0 else batch_losses[loss]['val']
                        loss_vals = loss_vals + ', {}: {:.8f}'.format(loss, mean_loss)
                self.logger.debug('{} epoch: {} [{}/{} ({:.0f}%)]{}'.format(
                        mode, epoch, (batch_idx + 1), num_batches,
                        100.0 * (batch_idx + 1.0) / num_batches, loss_vals))

            # Populate epoch losses
            for k, v in batch_losses.items():
                if k in epoch_losses:
                    epoch_losses[k]['val'] += v['val']
                    epoch_losses[k]['numel'] += v['numel']
                else:
                    epoch_losses[k] = v

            # Save preds to files
            if batch_debug:
                save_dict = {}
                debug_dir = os.path.join(self.config['outdir'], 'debug')
                create_directory(debug_dir)
                save_path = os.path.join(debug_dir, 'epoch{}_batch{}.npz'.format(epoch, batch_idx))

                # Store source, target, smask and tmask per batch element
                for i in range(len(batch)):
                    save_dict[str(i) + '_source'] = np.copy(batch[i][0].detach().cpu().numpy())
                    save_dict[str(i) + '_target'] = np.copy(batch[i][1].detach().cpu().numpy())
                    save_dict[str(i) + '_smask'] = np.copy(batch[i][2].detach().cpu().numpy())
                    save_dict[str(i) + '_tmask'] = np.copy(batch[i][3].detach().cpu().numpy())
                # Store debug data
                for k, v in batch_debug.items():                    
                    if type(v) in [list, tuple]: # assume that v has items per-batch-element
                        assert len(v) == len(batch), "Only per-batch-element lists are allowed for debug tensors"
                        for i in range(len(v)):
                            try: # v[i] is tensor
                                save_dict[str(i) + '_' + k] = np.copy(v[i].detach().cpu().numpy())
                            except: # v[i] is not a tensor
                                save_dict[str(i) + '_' + k] = v[i]
                    else: # assume a single item for the whole batch
                        try: # v is a tensor
                            save_dict[k] = np.copy(v.detach().cpu().numpy())
                        except: # v is not a tensor
                            save_dict[k] = v
                np.savez_compressed(save_path, **save_dict)

        # Return epoch records
        epoch_records = {}
        for k, v in epoch_losses.items():
            epoch_records[k] = v['val'] if v['numel'] == 0. else v['val'] / float(v['numel'])
        return epoch_records

    def evaluate(self, data_loader, *args, **kwargs):
        # Run eval
        t = time.time()
        epoch_records = self.run_epoch(data_loader, 'eval', 0, *args, **kwargs)

        # Log and print epoch records
        log_record_dict('Eval', epoch_records, self.global_records)
        self.print_record_dict(epoch_records, 'Eval', time.time() - t)
        self.global_records['result'].update({
            'loss': epoch_records['loss'],
        })

    def save_net(self, filename):
        torch.save(self.net.state_dict(), filename)
        self.logger.info('params saved to {}'.format(filename))

    def load_net(self, filename):
        self.logger.info('Loading params from {}'.format(filename))
        self.net.load_state_dict(torch.load(filename), strict=False)

    def print_record_dict(self, record_dict, usage, t_taken):
        loss_str = ''
        for k, v in record_dict.items():
            loss_str = loss_str + ' {}: {:.8f}'.format(k, v)
        self.logger.info('{}:{} took {:.3f}s'.format(
                usage, loss_str, t_taken))

    def _plot_helper(self, record_dict):
        plots = {}
        for loss in record_dict.keys():
            plots[loss] = {
                'train': get_records('train.' + loss, self.global_records),
                'valid': get_records('valid.' + loss, self.global_records)
            }
        return plots

    def get_burn_in_steps(self, seq_length, mode='train'):
        """ Linearly decrease burn_in_steps each epoch (if dynamic_burn_in allowed) """
        burn_in_steps = self.config[mode].get('burn_in_steps', -1)
        dynamic_burn_in = self.config[mode].get('dynamic_burn_in', mode=='train')
        if dynamic_burn_in and burn_in_steps > 0:
            burn_in_steps = max(burn_in_steps, seq_length - self.epoch)

        return burn_in_steps
