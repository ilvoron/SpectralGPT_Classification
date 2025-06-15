import argparse
import datetime
import json
import numpy as np
import os
import time
import wandb
from pathlib import Path

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
import util.lr_decay as lrd
import util.misc as misc
from util.datasets import build_fmow_dataset
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
import models_vit_tensor
import warnings

warnings.filterwarnings("ignore")
from engine_finetune import (train_one_epoch, evaluate)


def get_args_parser():
    parser = argparse.ArgumentParser('Тонкая настройка для классификации изображений', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Размер батча на GPU (эффективный размер батча = batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Количество итераций накопления градиентов (для увеличения эффективного размера батча при ограничениях по памяти)')

    parser.add_argument('--input_size', default=224, type=int,
                        help='Размер входных изображений')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Размер патча изображений')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Вероятность Drop path (по умолчанию: 0.1)')

    # Параметры оптимизатора
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Ограничение нормы градиента (по умолчанию: None, без ограничения)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (по умолчанию: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='Абсолютное значение learning rate')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='Базовый learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='Понижение learning rate по слоям (layer-wise lr decay) из ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-14, metavar='LR',
                        help='Нижняя граница learning rate для циклических scheduler-ов, которые достигают 0')

    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='Количество эпох для разогрева learning rate')

    # Параметры аугментации
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Фактор цветового дрожания (включается только если не используется Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Использовать политику AutoAugment. "v0" или "original". (по умолчанию: rand-m9-mstd0.5-inc1)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (по умолчанию: 0.1)')

    # * Параметры Random Erase
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Вероятность Random erase (по умолчанию: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Режим Random erase (по умолчанию: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Количество Random erase (по умолчанию: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Не применять Random erase к первой (чистой) аугментации')

    # * Параметры Mixup
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup включён если > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix включён если > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='Минимальное/максимальное соотношение cutmix, переопределяет alpha и включает cutmix, если задано (по умолчанию: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Вероятность применения mixup или cutmix, если включён хотя бы один')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Вероятность переключения на cutmix, если включены оба')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='Как применять параметры mixup/cutmix: "batch", "pair" или "elem"')

    # * Параметры тонкой настройки
    parser.add_argument('--finetune', default='',
                        help='Тонкая настройка из чекпойнта')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Использовать class token вместо global pool для классификации')

    # Параметры датасета
    parser.add_argument('--train_path', default='./txt_file/train_euro_result.txt', type=str,
                        help='Путь к обучающему .csv')
    parser.add_argument('--test_path', default='./txt_file/val_euro_result.txt', type=str,
                        help='Путь к тестовому .csv')
    parser.add_argument('--dataset_type', default='rgb',
                        choices=['rgb', 'temporal', 'sentinel', 'euro_sat', 'naip', 'bigearthnet', 'bigearthnet_finetune'],
                        help='Использовать fmow rgb, sentinel или другой датасет.')
    parser.add_argument('--masked_bands', default=None, nargs='+', type=int,
                        help='Последовательность индексов каналов для маскирования (средним значением) в датасете sentinel')
    parser.add_argument('--dropped_bands', type=int, nargs='+', default=None,
                        help="Какие каналы (с 0) удалить из данных sentinel.")

    parser.add_argument('--nb_classes', default=62, type=int,
                        help='Количество классов для классификации')

    parser.add_argument('--output_dir', default='./experiments/finetune',
                        help='Путь для сохранения, пусто — не сохранять')
    parser.add_argument('--log_dir', default='./experiments/finetune',
                        help='Путь для логов tensorboard')
    parser.add_argument('--device', default='cuda',
                        help='Устройство для обучения/тестирования')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='Возобновить из чекпойнта')
    parser.add_argument('--save_every', type=int, default=1, help='Как часто (в эпохах) сохранять чекпойнт')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help="Имя проекта Wandb, например: eurosat_finetune")
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="Имя пользователя Wandb, например: eurosat_finetune_user")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='Начальная эпоха')
    parser.add_argument('--eval', action='store_true',
                        help='Только оценка')
    parser.add_argument('--dist_eval', action='store_true', default=True,
                        help='Включить распределённую оценку (рекомендуется для быстрого мониторинга во время обучения)')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Закрепить память CPU в DataLoader для более эффективной (иногда) передачи на GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # Параметры распределённого обучения
    parser.add_argument('--world_size', default=1, type=int,
                        help='Количество распределённых процессов')
    parser.add_argument('--local-rank', default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL для настройки распределённого обучения')

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('рабочая директория: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train = build_fmow_dataset(is_train=True, args=args)
    dataset_val = build_fmow_dataset(is_train=False, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Внимание: Включена распределённая оценка с тестовым датасетом, не делящимся нацело на количество процессов. '
                      'Это немного изменит результаты валидации, так как будут добавлены дублирующиеся элементы для выравнивания '
                      'числа примеров на процесс.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank,
                shuffle=True)  # shuffle=True для уменьшения bias мониторинга
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=8,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)

    model = models_vit_tensor.__dict__["vit_base_patch8_128"](drop_path_rate=args.drop_path, num_classes=args.nb_classes)

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)

        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()

        for k in ['patch_embed.0.proj.weight', 'patch_embed.1.proj.weight', 'patch_embed.2.proj.weight',
                  'patch_embed.2.proj.bias', 'head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        if args.global_pool:
            print(set(msg.missing_keys))
        else:
            print(set(msg.missing_keys))

        trunc_normal_(model.head.weight, std=2e-5)  # вручную инициализируем слой fc

    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Модель = %s" % str(model_without_ddp))
    print('Количество параметров (М): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # задан только base_lr
        args.lr = args.blr * eff_batch_size / 256

    print("Базовый lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("Текущий lr: %.2e" % args.lr)

    print("Итераций накопления градиента: %d" % args.accum_iter)
    print("Эффективный размер батча: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
                                        no_weight_decay_list=model_without_ddp.no_weight_decay(),
                                        layer_decay=args.layer_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    if mixup_fn is not None:
        if args.dataset_type != 'bigearthnet_finetune':
            # сглаживание реализовано через преобразование меток mixup
            criterion = SoftTargetCrossEntropy()
        if args.dataset_type == 'bigearthnet_finetune':
            criterion = torch.nn.MultiLabelSoftMarginLoss()
    else:
        if args.dataset_type != 'bigearthnet_finetune':
            # сглаживание реализовано через преобразование меток mixup
            criterion = SoftTargetCrossEntropy()
        if args.dataset_type == 'bigearthnet_finetune':
            criterion = torch.nn.MultiLabelSoftMarginLoss()

    print("Критерий = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0 and args.wandb_project is not None and args.wandb_entity is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity)
        wandb.config.update(args)
        wandb.watch(model)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Оценка на {len(dataset_val)} тестовых изображениях - acc1: {test_stats['acc1']:.2f}%, acc5: {test_stats['acc5']:.2f}%")
        exit(0)

    print(f"Начало обучения на {args.epochs} эпохах")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, mixup_fn,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_every == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch >= 0:
            test_stats = evaluate(data_loader_val, model, device, args)

        if args.dataset_type == 'bigearthnet_finetune':
            if epoch >= 0:
                print(f"Точность сети на {len(dataset_val)} тестовых изображениях: {test_stats['mAP']:.1f}%")
                if log_writer is not None:
                    log_writer.add_scalar('perf/test_mAP', test_stats['mAP'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

        if args.dataset_type != 'bigearthnet_finetune':
            if epoch >= 0:
                print(f"Точность сети на {len(dataset_val)} тестовых изображениях: {test_stats['acc1']:.1f}%")
                max_accuracy = max(max_accuracy, test_stats["acc1"])
                print(f'Максимальная точность: {max_accuracy:.2f}%')

                if log_writer is not None:
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
        if epoch < 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
        if epoch >= 0:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

            if args.wandb is not None:
                try:
                    wandb.log(log_stats)
                except ValueError:
                    print(f"Некорректная статистика?")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Время обучения {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
