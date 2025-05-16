import time
import copy
import os
from os.path import join
import datetime as dt
import sys
from os.path import dirname, abspath, join, basename, expanduser, normpath
from typing import List
from collections import Counter

import tqdm
from torch import optim
import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler


import numpy as np
import yaml
import pickle as pkl

root_dir = dirname((abspath(__file__)))
sys.path.append(root_dir)

import constants
from base.trainer import GenericVideoTrainer
from base.scheduler import GradualWarmupScheduler
from base.scheduler import MyWarmupScheduler

from instantiators import get_optimizer_scheduler
import dllogger as DLLogger
from tools import fmsg
from reproducibility import set_seed
from metrics import compute_f1_score
from metrics import compute_class_acc
from metrics import compute_confusion_matrix
from metrics import format_trg_pred_video
from metrics import format_trg_pred_frames
from metrics import PerfTracker
from abaw5_pre_processing.dlib.utils.shared import move_state_dict_to_device
from bah_metrics import bah_perfs


def count_params(params) -> int:
    return sum([p.numel() for p in params])


class Trainer(GenericVideoTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.best_epoch_info = {
            'model_weights': copy.deepcopy(self.model.state_dict()),
            'loss': 1e10,
            'ccc': -1e10,
            'ce': 10.,
            'acc': -1,
            'p_r_f1': 0,
            'kappa': 0,
            'epoch': 0,
            'metrics': {
                'train_loss': -1,
                'val_loss': -1,
                'train_acc': -1,
                'val_acc': -1,
            }
        }

        self.args = None

        self.epoch = 0
        self.counter = 0
        self.seed = 0
        self.default_seed = None
        self.max_seed = (2 ** 32) - 1
        msg = f"seed must be: 0 <= {self.seed} <= {self.max_seed}"
        assert 0 <= self.seed <= self.max_seed, msg

        self.dataloaders = None
        self.number_classes = None

        self.t_init_epoch = dt.datetime.now()
        self.t_end_epoch = dt.datetime.now()
        self.cl_to_int: dict = dict()
        self.int_to_cl: dict = dict()

    def set_number_classes(self, ncls: int):
        assert ncls > 0, ncls
        assert isinstance(ncls, int), type(ncls)

        self.number_classes = ncls

    def set_args(self, args):
        self.args = args

    def post_set_args(self):
        assert self.args is not None
        fold = self.args.fold_to_run
        folds_path = join(root_dir, self.args.folds_dir,
                          f"split-{fold}")
        path_class_id = join(folds_path, 'class_id.yaml')
        with open(path_class_id, 'r') as fcl:
            cl_int = yaml.safe_load(fcl)

        self.cl_to_int: dict = cl_int
        self.int_to_cl: dict = self.switch_key_val_dict(cl_int)

    @staticmethod
    def switch_key_val_dict(d: dict) -> dict:
        out = dict()
        for k in d:
            assert d[k] not in out, 'more than 1 key with same value. wrong.'
            out[d[k]] = k

        return out

    def init_seed(self):
        assert self.args is not None

        self.epoch = 0
        self.counter = 0
        self.seed = int(self.args.seed)
        self.default_seed = int(self.args.seed)
        self.max_seed = (2 ** 32) - 1
        msg = f"seed must be: 0 <= {self.seed} <= {self.max_seed}"
        assert 0 <= self.seed <= self.max_seed, msg

    def init_optimizer_and_scheduler(self, epoch=0):

        params = self.get_parameters()
        DLLogger.log(f"There are: {count_params(params)} params to be updated.")

        self.optimizer, self.scheduler = get_optimizer_scheduler(
            vars(self.args), params, epoch,
            best=self.best_epoch_info['ce']
        )

        # old version
        # self.optimizer = optim.Adam(self.get_parameters(),
        #                             lr=self.learning_rate,
        #                             weight_decay=0.001
        #                             )
        #
        # self.scheduler = MyWarmupScheduler(
        #     optimizer=self.optimizer,
        #     lr=self.learning_rate,
        #     min_lr=self.min_learning_rate,
        #     best=self.best_epoch_info['ce'],
        #     mode="max",
        #     patience=self.patience,
        #     factor=self.factor,
        #     num_warmup_epoch=self.min_epoch,
        #     init_epoch=epoch
        # )

    def fit(self, dataloader_dict, checkpoint_controller, parameter_controller):

        if self.verbose:
            print("------")
            print("Starting training, on device:", self.device)

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }

        for epoch in np.arange(start_epoch, self.max_epoch):

            if self.fit_finished:
                if self.verbose:
                    print("\nEarly Stop!\n")
                break

            improvement = False

            cnd = (parameter_controller.get_current_lr() <
                   self.min_learning_rate
                   )
            cnd &= (epoch >= self.min_epoch)
            cnd &= (self.scheduler.relative_epoch > self.min_epoch)

            if epoch in self.milestone or cnd:
                parameter_controller.release_param(self.model.spatial, epoch)
                if parameter_controller.early_stop:
                    break

                self.model.load_state_dict(self.best_epoch_info['model_weights'])

            time_epoch_start = time.time()

            if self.verbose:
                zz = len(self.optimizer.param_groups[0]['params'])
                print(f"There are {zz} layers to update.")

            # Get the losses and the record dictionaries for training and
            # validation.
            train_kwargs = {"dataloader_dict": dataloader_dict, "epoch": epoch}
            train_loss, train_record_dict = self.train(**train_kwargs)

            validate_kwargs = {"dataloader_dict": dataloader_dict,
                               "epoch": epoch}
            validate_loss, validate_record_dict = self.validate(
                **validate_kwargs)

            # if epoch % 1 == 0:
            #     test_kwargs = {"dataloader_dict": dataloader_dict,
            #     "epoch": None, "train_mode": 0}
            #     validate_loss, test_record_dict = self.test(
            #     checkpoint_controller=checkpoint_controller,
            #     feature_extraction=0, **test_kwargs)
            #     print(test_record_dict['overall']['ccc'])

            if validate_loss < 0:
                raise ValueError('validate loss negative')

            self.train_losses.append(train_loss)
            self.validate_losses.append(validate_loss)

            validate_ccc = validate_record_dict['overall']['ccc']

            self.scheduler.best = self.best_epoch_info['ccc']


            if validate_ccc > self.best_epoch_info['ccc']:
                torch.save(self.model.state_dict(),
                           join(self.save_path,
                                f"model_state_dict{validate_ccc}.pth"))

                improvement = True
                self.best_epoch_info = {
                    'model_weights': copy.deepcopy(self.model.state_dict()),
                    'loss': validate_loss,
                    'ccc': validate_ccc,
                    'epoch': epoch,
                }

            if self.verbose:
                print(
                    "\n Fold {:2} Epoch {:2} in {:.0f}s || Train loss={:.3f} | Val loss={:.3f} | LR={:.1e} | Release_count={} | best={} | "
                    "improvement={}-{}".format(
                        self.fold,
                        epoch + 1,
                        time.time() - time_epoch_start,
                        train_loss,
                        validate_loss,
                        self.optimizer.param_groups[0]['lr'],
                        parameter_controller.release_count,
                        int(self.best_epoch_info['epoch']) + 1,
                        improvement,
                        self.early_stopping_counter))

                print(train_record_dict['overall'])
                print(validate_record_dict['overall'])
                print("------")

            checkpoint_controller.save_log_to_csv(
                epoch, train_record_dict['overall'],
                validate_record_dict['overall'])

            # Early stopping controller.
            cnd = (self.scheduler.relative_epoch > self.min_epoch)
            if self.early_stopping and cnd:
                if improvement:
                    self.early_stopping_counter = self.early_stopping
                else:
                    self.early_stopping_counter -= 1

                if self.early_stopping_counter <= 0:
                    self.fit_finished = True

            self.scheduler.step(metrics=validate_ccc, epoch=epoch)


            self.start_epoch = epoch + 1

            if self.load_best_at_each_epoch:
                self.model.load_state_dict(
                    self.best_epoch_info['model_weights'])

            checkpoint_controller.save_checkpoint(
                self, parameter_controller, self.save_path)

        self.fit_finished = True
        checkpoint_controller.save_checkpoint(self, parameter_controller,
                                              self.save_path)

        self.model.load_state_dict(self.best_epoch_info['model_weights'])

    def random(self):
        self.counter = self.counter + 1
        seed = self.default_seed + self.counter
        self.seed = int(seed % self.max_seed)
        set_seed(seed=self.seed, verbose=False)

    def default_random(self):
        set_seed(self.default_seed)

    def on_epoch_start(self):
        self.t_init_epoch = dt.datetime.now()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

    def on_epoch_end(self):
        self.t_end_epoch = dt.datetime.now()
        delta_t = self.t_end_epoch - self.t_init_epoch
        c_ep = self.counter
        t_ep = self.args.num_epochs
        DLLogger.log(fmsg(f'Train epoch ({c_ep - 1}/{t_ep}) runtime:'
                          f' {delta_t}'))

    def train_one_epoch(self):
        self.random()
        self.on_epoch_start()
        self.model.train()

        dataloader = self.dataloaders[constants.TRAINSET]

        # dataloader_dict, epoch, train_mode = kwargs['dataloader_dict'], kwargs[
        #     'epoch'], kwargs['train_mode']


        running_loss = 0.0
        total_batch_counter = 0
        inputs = {}

        # output_handler = ContinuousOutputHandler()
        # continuous_label_handler = ContinuousOutputHandler()
        #
        # metric_handler = ContinuousMetricsCalculator(self.metrics,
        #                                              self.emotion,
        #                                              output_handler,
        #                                              continuous_label_handler
        #                                              )

        num_batch_warm_up = len(dataloader) * self.min_epoch

        scaler = GradScaler(enabled=self.args.amp)

        count = 0

        for batch_idx, (X, trials, lengths, indices) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), ncols=80):
            # trials: list of video ids.

            total_batch_counter += len(trials)

            for feature, value in X.items():
                inputs[feature] = X[feature].to(self.device)

            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)  # bsz, nframes, 1

            elif constants.EXPR in inputs:  # EXPR_continuous_label
                labels = inputs.pop(constants.EXPR, None)

            else:
                raise NotImplementedError

            global_sup = inputs.pop('global_sup')
            assert global_sup.shape == labels.shape, f"{global_sup.shape} | " \
                                                     f"{labels.shape}"
            # print(labels.shape, 'train labels')
            # todo : fix this.
            if len(torch.flatten(labels)) == self.train_batch_size:
                labels = torch.zeros((self.train_batch_size,
                                      len(indices[0]), 1),
                                     dtype=torch.float32).to(self.device)
                DLLogger.log('Error - train - fix it.')

                sys.exit()

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.args.amp):
                outputs = self.model(inputs)  # bsz, nfames, ncls

                bsz, nfms, d = labels.shape  # float32.
                assert d == 1, d
                _labels = labels.contiguous().view(bsz * nfms).long()

                assert outputs.ndim == 3, outputs.ndim
                ncls = self.number_classes
                assert outputs.shape[0] == bsz, f"{outputs.shape[0]} | {bsz}"
                assert outputs.shape[1] == nfms, f"{outputs.shape[1]} | {nfms}"
                assert outputs.shape[2] == ncls, f"{outputs.shape[2]} | {ncls}"

                _labels = labels.contiguous().view(bsz * nfms).long()
                _global_sup = global_sup.contiguous().view(bsz * nfms).long()
                _outputs = outputs.contiguous().view(bsz * nfms, ncls)

                if self.args.train_supervision_type == \
                        constants.TR_SUP_VIDEO_GL:
                    tr_superv = _global_sup

                elif self.args.train_supervision_type == \
                        constants.TR_SUP_VIDEO_FR:
                    tr_superv = _labels

                else:
                    raise NotImplementedError(self.args.train_supervision_type)

                loss = self.criterion(_outputs, tr_superv)

            running_loss = running_loss + loss.mean().detach()
            count = count + 1

            # if train_mode:
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # output_handler.update_output_for_seen_trials(
            #     outputs.detach().cpu().numpy(), trials, indices, lengths)
            # continuous_label_handler.update_output_for_seen_trials(
            #     labels.detach().cpu().numpy(), trials, indices,
            #     lengths)

        epoch_loss = running_loss / float(count)

        # output_handler.average_trial_wise_records()
        # continuous_label_handler.average_trial_wise_records()
        #
        # output_handler.concat_records()
        # continuous_label_handler.concat_records()

        # Compute the root mean square error, pearson correlation coefficient
        # and significance, and the
        # concordance correlation coefficient.
        # They are calculated by  first concatenating all the output
        # and continuous labels to two long arrays, and then calculate the
        # metrics.
        # metric_handler.calculate_metrics()
        # epoch_result_dict = metric_handler.metric_record_dict
        #
        # metric_handler.save_trial_wise_records(self.save_path, train_mode,
        #                                        epoch)

        if self.save_plot:
            pass
            # This object plot the figures and save them.
            # plot_handler = PlotHandler(self.metrics, self.emotion,
            #                            epoch_result_dict,
            #                            output_handler.trialwise_records,
            #                            continuous_label_handler.trialwise_records,
            #                            epoch=epoch, train_mode=train_mode,
            #                            directory_to_save_plot=self.save_path)
            # plot_handler.save_output_vs_continuous_label_plot()

        epoch_result_dict = None

        self.on_epoch_end()

        return epoch_loss.item()

    def inference(self, dataloader, split: str):
        self.default_random()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        self.model.eval()
        n_videos = 0
        inputs = {}

        per_video_frame_logits = {}

        for batch_idx, (X, trials, lengths, indices) in tqdm.tqdm(
                enumerate(dataloader), total=len(dataloader), ncols=80):
            # trials: list of video ids.

            n_videos = n_videos + len(trials)

            nframes = 0
            for feature, value in X.items():
                inputs[feature] = X[feature].to(self.device)
                bsz = inputs[feature].shape[0]
                assert bsz == 1, f"{bsz} | {feature} | {batch_idx}"

                if feature == constants.EXPR:
                    nframes = value.shape[1]

                # video torch.Size([1, 300, 3, 40, 40]) ['test/dia12_utt6']
                # vggish torch.Size([1, 1, 300, 128]) ['test/dia12_utt6']
                # bert torch.Size([1, 1, 300, 768]) ['test/dia12_utt6']
                # EXPR_continuous_label torch.Size([1, 300,
                # 1]) ['test/dia12_utt6']

            assert nframes > 0, nframes

            if "continuous_label" in inputs:
                labels = inputs.pop("continuous_label", None)  # bsz, nframes, 1

            elif constants.EXPR in inputs:  # EXPR_continuous_label
                labels = inputs.pop(constants.EXPR, None)
            else:
                raise NotImplementedError

            global_sup = inputs.pop('global_sup')
            assert global_sup.shape == labels.shape, f"{global_sup.shape} | " \
                                                     f"{labels.shape}"

            # print(labels.shape, global_sup.shape, 'label inference')
            # todo : fix this.
            if len(torch.flatten(labels)) == self.train_batch_size:
                labels = torch.zeros((self.train_batch_size,
                                      len(indices[0]), 1),
                                     dtype=torch.float32).to(self.device)
                DLLogger.log('Error - eval - fix it.')
                sys.exit()

            with autocast(enabled=self.args.amp):
                with torch.no_grad():
                    cnd = (nframes > self.args.window_length)
                    cnd &= (self.args.model_name == constants.LFAN)
                    if cnd:
                        outputs = self.inference_forward_windows(inputs)
                    else:
                        outputs = self.model(inputs)  # bsz, nfames, ncls

                    outputs = outputs.detach()

                bsz, nfms, d = labels.shape
                assert d == 1, d
                _labels = labels.contiguous().view(bsz * nfms).long()
                _global_sup = global_sup.contiguous().view(bsz * nfms).long()
                assert outputs.ndim == 3, outputs.ndim
                ncls = self.number_classes
                assert outputs.shape[0] == bsz, f"{outputs.shape[0]} | {bsz}"
                assert outputs.shape[1] == nfms, f"{outputs.shape[1]} | {nfms}"
                assert outputs.shape[2] == ncls, f"{outputs.shape[2]} | {ncls}"

                _labels = labels.contiguous().view(bsz * nfms).long()
                _global_sup = global_sup.contiguous().view(bsz * nfms).long()
                _outputs = outputs.contiguous().view(bsz * nfms, ncls)

                _v_id = trials[0]
                # assumes no windowing.
                assert _v_id is not per_video_frame_logits, _v_id
                per_video_frame_logits[_v_id] = {
                    'labels': _labels.cpu().numpy().flatten(),
                    'logits': _outputs.detach().cpu().numpy(),
                    'g_labels': _global_sup.cpu().numpy().flatten(),
                }

        # performance evaluation.

        current_perf = self.compute_perf(per_video_frame_logits)
        # store if necessary:
        # if self.args.dataset_name in [constants.C_EXPR_DB_CHALLENGE,
        #                               constants.AH_DB, constants.BAH_DB]:
        #     out_inf = self.args.outd
        #     os.makedirs(out_inf, exist_ok=True)
        #     f_preds = join(out_inf, f'prediction-{split}.pkl')
        #     with open(f_preds, 'wb') as fxx:
        #         pkl.dump(per_video_frame_logits, fxx,
        #                  protocol=pkl.HIGHEST_PROTOCOL)
        #     print(f"Dumps the predictions of {self.args.dataset_name} "
        #           f"at {f_preds}")

        return current_perf, per_video_frame_logits

    def compute_perf(self, data) -> dict:
        # data: dict where each key is a video. see self.inference
        frame_lbls = None
        video_lbls = None
        frame_logits = None
        for k in data:
            if frame_lbls is None:
                frame_lbls = data[k]['labels']
                frame_logits = data[k]['logits']
                video_lbls = data[k]['g_labels']

            else:
                frame_lbls = np.concatenate((frame_lbls, data[k]['labels']),
                                            axis=0)
                frame_logits = np.concatenate((frame_logits, data[k]['logits']),
                                              axis=0)
                video_lbls = np.concatenate((video_lbls, data[k]['g_labels']),
                                            axis=0)

        # frame performance
        frame_preds = np.argmax(frame_logits, axis=1)
        frame_perfs = bah_perfs(gt=frame_lbls,
                              hard_preds=frame_preds,
                              logits=frame_logits
                              )

        return frame_perfs

    def _old_compute_perf(self, data) -> dict:
        # frame level performance
        perf = dict()

        _atom = {'master': 0.0, 'per_cl': 0.0}
        _video = dict()
        for k in constants.VIDEO_PREDS:
            _video[k] = copy.deepcopy(_atom)

        for mtr in constants.METRICS:

            perf[mtr] = {
                constants.FRAME_LEVEL: copy.deepcopy(_atom),
                constants.VIDEO_LEVEL: copy.deepcopy(_video),
            }

        all_perf = dict()
        l_ignore_class = [None]
        if (self.args.dataset_name == constants.C_EXPR_DB) and (
            self.args.use_other_class
        ):
            _other_int = self.cl_to_int[constants.OTHER]
            assert _other_int == 7, _other_int
            l_ignore_class.append(_other_int)  # 'Other' class


        for ignore_class in l_ignore_class:
            _perf = copy.deepcopy(perf)

            # formatting frame level
            preds, trgs = format_trg_pred_frames(data,
                                                 ignore_class=ignore_class)

            f1_per_cl, macro_f1 = compute_f1_score(trgs, preds,
                                                   constants.MACRO_F1)
            _, w_f1 = compute_f1_score(trgs, preds, constants.W_F1)

            acc = compute_class_acc(trgs, preds)
            cnf_mtx = compute_confusion_matrix(trgs, preds)

            _perf[constants.MACRO_F1][constants.FRAME_LEVEL] = {
                'master': macro_f1, 'per_cl': f1_per_cl
            }
            _perf[constants.W_F1][constants.FRAME_LEVEL] = {
                'master': w_f1, 'per_cl': f1_per_cl
            }

            _perf[constants.CL_ACC][constants.FRAME_LEVEL] = {
                'master': acc, 'per_cl': acc  # just avg
            }
            _perf[constants.CFUSE_MARIX][constants.FRAME_LEVEL] = {
                'master': cnf_mtx, 'per_cl': cnf_mtx
            }

            # formatting video level
            preds, trgs = format_trg_pred_video(data, ignore_class=ignore_class)

            for k in preds[0]:
                _preds_k = [item[k] for item in preds]
                f1_per_cl, macro_f1 = compute_f1_score(trgs, _preds_k,
                                                       constants.MACRO_F1)
                _, w_f1 = compute_f1_score(trgs, _preds_k, constants.W_F1)
                acc = compute_class_acc(trgs, _preds_k)
                cnf_mtx = compute_confusion_matrix(trgs, _preds_k)

                _perf[constants.MACRO_F1][constants.VIDEO_LEVEL][k] = {
                    'master': macro_f1, 'per_cl': f1_per_cl
                }
                _perf[constants.W_F1][constants.VIDEO_LEVEL][k] = {
                    'master': w_f1, 'per_cl': f1_per_cl
                }

                _perf[constants.CL_ACC][constants.VIDEO_LEVEL][k] = {
                    'master': acc, 'per_cl': acc
                }
                _perf[constants.CFUSE_MARIX][constants.VIDEO_LEVEL][k] = {
                    'master': cnf_mtx, 'per_cl': cnf_mtx
                }

            all_perf[ignore_class] = copy.deepcopy(_perf)

        return all_perf

    @property
    def cpu_device(self):
        return torch.device("cpu")

    def optimize(self,
                 dataloader_dict,
                 checkpoint_controller=None,
                 parameter_controller=None
                 ):

        self.dataloaders = dataloader_dict

        DLLogger.log(fmsg(f"Starting training"))

        self.init_seed()
        self.random()

        self.time_fit_start = time.time()
        start_epoch = self.start_epoch

        if self.best_epoch_info is None:
            self.best_epoch_info = {
                'model_weights': copy.deepcopy(self.model.state_dict()),
                'loss': 1e10,
                'ccc': -1e10
            }

        current_perf, _ = self.inference(self.dataloaders[constants.VALIDSET],
                                         constants.VALIDSET)

        best_model = copy.deepcopy(self.model).to(self.cpu_device).eval()
        best_perf = -1
        best_epoch = -1

        master_metric = constants.AP_POS

        valid_tracker = []
        mtrs = [constants.W_F1, constants.CL_ACC, constants.F1_POS,
                constants.AP_POS]

        for epoch in tqdm.tqdm(np.arange(0, self.max_epoch), ncols=80,
                               total=self.max_epoch):

            epoch_loss = self.train_one_epoch()

            self.scheduler.step()

            # validation:
            _split = constants.VALIDSET
            current_perf, per_video_frame_logits = self.inference(
                self.dataloaders[constants.VALIDSET], _split)

            valid_tracker.append(current_perf)


            if current_perf[master_metric] >= best_perf:
                best_model = copy.deepcopy(self.model).to(
                    self.cpu_device).eval()
                best_perf = current_perf[master_metric]
                best_epoch = epoch

                # dump the predictions
                _out_inf = self.args.outd
                os.makedirs(_out_inf, exist_ok=True)
                f_preds = join(_out_inf, f'prediction-{_split}.pkl')
                with open(f_preds, 'wb') as fxx:
                    pkl.dump(per_video_frame_logits, fxx,
                             protocol=pkl.HIGHEST_PROTOCOL)

                # dump performance

                current_perf[constants.CFUSE_MARIX] = current_perf[
                    constants.CFUSE_MARIX].tolist()

                _data = {
                    'dataset': self.args.dataset_name,
                    'split': _split,
                    'perfs_frames': current_perf
                }
                with open(join(_out_inf, f'perf-{_split}.yml'),
                          'w') as fx:
                    yaml.dump(_data, fx)

            msg = f'Split [{_split}]: '
            for mtr in mtrs:
                if mtr == master_metric:
                    msg += f"[{mtr}]:{current_perf[mtr]:.4f}|{best_perf:.4f}*; "
                else:
                    msg += f"[{mtr}]:{current_perf[mtr]:.4f}; "

            msg += f"Best-epoch: {best_epoch}"

            DLLogger.log(msg)

        self.fit_finished = True

        self.args.best_epoch = best_epoch

        # test best model ======================================================

        DLLogger.log(fmsg(f"{constants.TESTSET} performance:"))

        _model = copy.deepcopy(best_model)
        _state_dict = _model.state_dict()  # cpu

        _state_dict = move_state_dict_to_device(_state_dict, self.device)
        self.model.load_state_dict(_state_dict, strict=True)

        _split = constants.TESTSET
        current_perf, per_video_frame_logits = self.inference(
            self.dataloaders[constants.TESTSET], _split)

        # dump the predictions
        _out_inf = self.args.outd
        os.makedirs(_out_inf, exist_ok=True)

        f_preds = join(_out_inf, f'prediction-{_split}.pkl')
        with open(f_preds, 'wb') as fxx:
            pkl.dump(per_video_frame_logits, fxx,
                     protocol=pkl.HIGHEST_PROTOCOL)

        # dump performance
        current_perf[constants.CFUSE_MARIX] = current_perf[
            constants.CFUSE_MARIX].tolist()

        _data = {
            'dataset': self.args.dataset_name,
            'split': _split,
            'perfs_frames': current_perf
        }
        with open(join(_out_inf, f'perf-{_split}.yml'),
                  'w') as fx:
            yaml.dump(_data, fx)

        msg = f'Split [{_split}]: '
        for mtr in mtrs:
            if mtr == master_metric:
                msg += f"[{mtr}]:{current_perf[mtr]:.4f}; "
            else:
                msg += f"[{mtr}]:{current_perf[mtr]:.4f}; "

        DLLogger.log(msg)
        # ======================================================================

        # store models weights.
        dir_best_model = join(self.args.outd, 'best-model')
        os.makedirs(dir_best_model, exist_ok=True)

        _model = copy.deepcopy(best_model)
        _state_dict = _model.state_dict()  # cpu

        torch.save(_state_dict, join(dir_best_model, 'model.pt'))
        path = join(dir_best_model, 'config.yml')
        self.save_args(path)

        self.args.tend = dt.datetime.now()

        path = join(self.args.outd, 'config.yml')
        self.save_args(path)

        self.bye(self.args)

    def save_args(self, path):
        _path = path
        with open(_path, 'w') as f:
            yaml.dump(vars(self.args), f)

    @staticmethod
    def bye(args):
        _args = copy.deepcopy(args)
        DLLogger.log(fmsg("End time: {}".format(_args.tend)))
        DLLogger.log(fmsg("Total time: {}".format(_args.tend - _args.t0)))

        with open(join(_args.outd, 'passed.txt'), 'w') as fout:
            fout.write('Passed.')

        DLLogger.log(fmsg('bye.'))

    def window_input(self, data: dict) -> list:
        sz = []
        # video torch.Size([1, 300, 3, 40, 40]) ['test/dia12_utt6']
        # vggish torch.Size([1, 1, 300, 128]) ['test/dia12_utt6']
        # bert torch.Size([1, 1, 300, 768]) ['test/dia12_utt6']
        for modality in data:

            _bsz = data[modality].shape[0]
            if modality == constants.VGGISH:
                _nfms = data[modality].shape[2]
            elif modality == constants.VIDEO:
                _nfms = data[modality].shape[1]
            elif modality == constants.BERT:
                _nfms = data[modality].shape[2]
            else:
                raise NotImplementedError(modality)

            sz.append([_bsz, _nfms])  # bsz, length

        for item in sz:
            assert item == sz[0], f"{item} | {sz[0]}"

        length = sz[0][1]
        windows = self.windowing(np.arange(length),
                                 self.args.window_length,
                                 self.args.hop_length
                                 )
        data_chunks = []
        for wd in windows:
            tmp = dict()
            for modality in data:
                if modality == constants.VGGISH:
                    tmp[modality] = data[modality][:, :, wd, ...]
                elif modality == constants.VIDEO:
                    tmp[modality] = data[modality][:, wd, ...]
                elif modality == constants.BERT:
                    tmp[modality] = data[modality][:, :, wd, ...]
                else:
                    raise NotImplementedError(modality)

            data_chunks.append([tmp, wd])

        return data_chunks

    def inference_forward_windows(self, data):
        # forward data that is longer than what the model can handle.
        # we window the input with a hope. forward each window. stichach back
        # the prediction to match the original input video.
        chunks = self.window_input(data)
        results = []
        nframes = 0
        total_frames = 0
        for modality in data:
            if modality == constants.VGGISH:
                total_frames = data[modality].shape[2]
            elif modality == constants.VIDEO:
                total_frames = data[modality].shape[1]
            elif modality == constants.BERT:
                total_frames = data[modality].shape[2]
            else:
                raise NotImplementedError(modality)

        for chunk in chunks:
            _data, wd = chunk
            _output = self.model(_data)  # bsz, nframes, ncls
            assert _output.ndim == 3, _output.ndim
            results.append([_output, wd])
            nframes = wd[-1]

        nframes = nframes + 1

        assert total_frames == nframes, f"{total_frames} | {nframes}"

        bsz = results[-1][0].shape[0]
        ncsl = results[-1][0].shape[2]

        final_out = torch.zeros((bsz, nframes, ncsl),
                                device=results[-1][0].device,
                                dtype=results[-1][0].dtype,
                                requires_grad=results[-1][0].requires_grad
                                )

        # stitch results
        idx = []
        for item in results:
            _output, wd = item
            final_out[:, wd, ...] = final_out[:, wd, ...] + _output
            idx = idx + wd.tolist()

        # average predictions where windows overlap.
        z = Counter(idx)
        z = z.most_common()  # list of tuples: [(idx, freq)]
        mixed = sorted(z, key=lambda x: x[0], reverse=False)
        indices = [item[0] for item in mixed]
        freqs = [item[1] for item in mixed]
        indices = np.asarray(indices, dtype=np.int64)
        freqs = torch.tensor(freqs,
                             dtype=final_out.dtype,
                             device=final_out.device,
                             requires_grad=False
                             )
        freqs = freqs.view(1, -1, 1)
        final_out[:, indices, ...] = final_out[:, indices, ...] / freqs

        return final_out

    @staticmethod
    def windowing(x, window_length, hop_length) -> List[np.ndarray]:
        _length = len(x)

        if _length >= window_length:
            steps = (_length - window_length) // hop_length + 1

            sampled_x = []
            for i in range(steps):
                start = i * hop_length
                end = start + window_length
                sampled_x.append(x[start:end])

            if sampled_x[-1][-1] < _length - 1:
                sampled_x.append(x[-window_length:])

        else:
            sampled_x = [x]

        return sampled_x