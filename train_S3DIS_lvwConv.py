import argparse
import time
import os
import datetime
import numpy as np
import random
import re
from datasets.S3DIS import S3DIStrain, cfl_collate_fn

# from datasets.S3DIS_rewrite_colortest import S3DIStrain, cfl_collate_fn
from tensorboardX import SummaryWriter
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.fpn_lvwConv import Res16FPN18_lvwConv_4_2_CSattn_2
from eval_S3DIS_lvwConv import eval
import textwrap
from lib.utils import get_pseudo, get_sp_feature, get_fixclassifier
from sklearn.cluster import KMeans
import logging
from os.path import join
import warnings

warnings.filterwarnings("ignore")
###
current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
default_save_path = f"ckpt/S3DIS_{current_date}_1/"
default_pseudo_label_path = f"pseudo_label_s3dis/{current_date}/"
# wandb.init(project="gpu_memory_tracking")


def parse_args():
    """PARAMETERS"""
    parser = argparse.ArgumentParser(description="PyTorch Unsuper_3D_Seg")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/data/S3DIS/input",
        help="pont cloud data path",
    )
    parser.add_argument(
        "--sp_path",
        type=str,
        default="/home/S3DIS/initial_superpoints/",
        help="initial superpoint path",
    )
    parser.add_argument(
        "--save_path", type=str, default=default_save_path, help="model savepath"
    )
    ###
    parser.add_argument(
        "--max_epoch",
        type=list,
        default=[500, 800],
        help="max epoch for non-growing and growing stage",
    )
    parser.add_argument(
        "--max_iter",
        type=list,
        default=[10000, 30000],
        help="max iter for non-growing and growing stage",
    )
    ###
    parser.add_argument(
        "--bn_momentum", type=float, default=0.02, help="batchnorm parameters"
    )
    parser.add_argument(
        "--conv1_kernel_size",
        type=int,
        default=5,
        help="kernel size of 1st conv layers",
    )
    ####
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD parameters")
    parser.add_argument("--dampening", type=float, default=0.1, help="SGD parameters")
    parser.add_argument(
        "--weight-decay", type=float, default=1e-4, help="SGD parameters"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="how many workers for loading data in training",
    )
    parser.add_argument(
        "--cluster_workers",
        type=int,
        default=4,
        help="how many workers for loading data in clustering",
    )
    parser.add_argument("--seed", type=int, default=2025, help="random seed")
    parser.add_argument("--log-interval", type=int, default=20, help="log interval")
    parser.add_argument(
        "--batch_size", type=int, default=10, help="batchsize in training"
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.05, help="voxel size in SparseConv"
    )
    parser.add_argument(
        "--input_dim", type=int, default=6, help="network input dimension"
    )  ### 6 for XYZGB
    parser.add_argument(
        "--primitive_num",
        type=int,
        default=300,
        help="how many primitives used in training",
    )
    parser.add_argument(
        "--semantic_class", type=int, default=12, help="ground truth semantic class"
    )
    parser.add_argument(
        "--feats_dim", type=int, default=128, help="output feature dimension"
    )
    parser.add_argument(
        "--pseudo_label_path",
        default=default_pseudo_label_path,
        type=str,
        help="pseudo label save path",
    )
    parser.add_argument("--ignore_label", type=int, default=12, help="invalid label")
    parser.add_argument(
        "--growsp_start",
        type=int,
        default=80,
        help="the start number of growing superpoint",
    )
    parser.add_argument(
        "--growsp_end",
        type=int,
        default=20,
        help="the end number of grwoing superpoint",
    )
    parser.add_argument(
        "--drop_threshold",
        type=int,
        default=10,
        help="ignore superpoints with few points",
    )
    parser.add_argument(
        "--w_rgb",
        type=float,
        default=5 / 5,
        help="weight for RGB in merging superpoint",
    )
    parser.add_argument(
        "--w_xyz",
        type=float,
        default=1 / 5,
        help="weight for XYZ in merging superpoint",
    )
    parser.add_argument(
        "--w_norm",
        type=float,
        default=4 / 5,
        help="weight for Normal in merging superpoint",
    )
    parser.add_argument(
        "--c_rgb", type=float, default=0, help="weight for RGB in clustering primitives"
    )
    parser.add_argument(
        "--c_shape",
        type=float,
        default=0,
        help="weight for PFH in clustering primitives",
    )
    return parser.parse_args()


def save_checkpoint_pair(model, classifier, save_path, epoch):
    ckpt_model = join(save_path, f"model_{epoch}_checkpoint.pth")
    ckpt_cls = join(save_path, f"cls_{epoch}_checkpoint.pth")
    torch.save(model.state_dict(), ckpt_model)
    torch.save(classifier.state_dict(), ckpt_cls)
    return ckpt_model, ckpt_cls


def manage_last_checkpoints(last_checkpoints, ckpt_pair, max_keep=5, keep_paths=None):
    keep_paths = keep_paths or set()
    last_checkpoints.append(ckpt_pair)

    while len(last_checkpoints) > max_keep:
        old_m, old_c = last_checkpoints[0]
        if old_m in keep_paths or old_c in keep_paths:
            last_checkpoints.pop(0)
        else:
            if os.path.exists(old_m):
                os.remove(old_m)
            if os.path.exists(old_c):
                os.remove(old_c)
            last_checkpoints.pop(0)
    return last_checkpoints


def manage_last_checkpoints(last_checkpoints, ckpt_pair, max_keep=5, keep=None):
    keep = keep or set()
    last_checkpoints.append(ckpt_pair)

    while len(last_checkpoints) > max_keep:
        old_m_path, old_c_path = last_checkpoints[0]
        if old_m_path in keep or old_c_path in keep:
            last_checkpoints.pop(0)
        else:
            if os.path.exists(old_m_path):
                os.remove(old_m_path)
            if os.path.exists(old_c_path):
                os.remove(old_c_path)
            last_checkpoints.pop(0)

    return last_checkpoints


def manage_best_record(
    miou_records,
    mIoU_value,
    oAcc_value,
    mAcc_value,
    ckpt_pair,
    last_checkpoints,
    best_oAcc_record=None,
    best_mAcc_record=None,
):

    miou_records.append((mIoU_value, *ckpt_pair))
    miou_records = sorted(miou_records, key=lambda x: x[0], reverse=True)
    keep = set()

    if len(miou_records) >= 1:
        keep.update([miou_records[0][1], miou_records[0][2]])  # 最高
    if len(miou_records) >= 2:
        keep.update([miou_records[1][1], miou_records[1][2]])  # 次高
    if len(miou_records) >= 3:
        keep.update([miou_records[-1][1], miou_records[-1][2]])  # 最低

    if best_oAcc_record is None or oAcc_value > best_oAcc_record[0]:
        best_oAcc_record = (oAcc_value, *ckpt_pair)
    keep.update(best_oAcc_record[1:])  # 加入oAcc关键路径
    if (best_oAcc_record[1], best_oAcc_record[2]) not in last_checkpoints:
        last_checkpoints.append((best_oAcc_record[1], best_oAcc_record[2]))

    if best_mAcc_record is None or mAcc_value > best_mAcc_record[0]:
        best_mAcc_record = (mAcc_value, *ckpt_pair)
    keep.update(best_mAcc_record[1:])  # 加入mAcc关键路径
    if (best_mAcc_record[1], best_mAcc_record[2]) not in last_checkpoints:
        last_checkpoints.append((best_mAcc_record[1], best_mAcc_record[2]))

    for m, m_path, c_path in miou_records[:]:
        if (
            m_path not in keep
            and c_path not in keep
            and (m_path, c_path) not in last_checkpoints
        ):
            if os.path.exists(m_path):
                os.remove(m_path)
            if os.path.exists(c_path):
                os.remove(c_path)
            miou_records.remove((m, m_path, c_path))

    return miou_records, best_oAcc_record, best_mAcc_record, last_checkpoints, keep


def run_training_stage(
    args,
    logger,
    train_loader,
    cluster_loader,
    model,
    optimizer,
    scheduler,
    loss,
    start_epoch,
    max_epoch,
    max_iter,
    is_Growing,
    last_checkpoints,
    miou_records,
):
    start_grow_epoch = start_epoch

    for epoch in range(1, max_epoch + 1):
        epoch += start_epoch

        if (epoch - 1) % 10 == 0:
            classifier, sp_s, p_s = cluster(
                args, logger, cluster_loader, model, epoch, start_grow_epoch, is_Growing
            )

        writer_loss, writer_epoch = train(
            train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier
        )
        writer.add_scalar("Training_Loss", writer_loss, writer_epoch)

        if epoch % 10 == 0:
            ckpt_pair = save_checkpoint_pair(model, classifier, args.save_path, epoch)

            with torch.no_grad():
                o_Acc, m_Acc, s = eval(epoch, args, ["Area_5"])
                logger.info(
                    f"Epoch: {epoch:02d}, oAcc {o_Acc:.2f} mAcc {m_Acc:.2f} IoUs{s}"
                )

            writer.add_scalar("Overall_Accuracy", o_Acc, epoch)
            writer.add_scalar("Mean_Accuracy", m_Acc, epoch)

            match = re.search(r"mIoU (\d+\.\d+)", s)
            if match:
                mIoU_value = float(match.group(1))
                writer.add_scalar("mIouU", mIoU_value, epoch)

                (
                    miou_records,
                    best_oAcc_record,
                    best_mAcc_record,
                    last_checkpoints,
                    keep,
                ) = manage_best_record(
                    miou_records, mIoU_value, o_Acc, m_Acc, ckpt_pair, last_checkpoints
                )

                last_checkpoints = manage_last_checkpoints(
                    last_checkpoints, ckpt_pair, max_keep=5, keep=keep
                )

            match = re.search(r"mIoU (\d+\.\d+)", sp_s)
            if match:
                writer.add_scalar("Superpoint_mIouU", float(match.group(1)), epoch)
            match = re.search(r"mIoU (\d+\.\d+)", p_s)
            if match:
                writer.add_scalar("Primitives_mIouU", float(match.group(1)), epoch)

            writer.flush()

            iterations = (epoch + 10) * len(train_loader)
            if iterations > max_iter:
                start_grow_epoch = epoch
                break

    return start_grow_epoch, last_checkpoints, miou_records


def main(args, logger):
    global writer
    writer = SummaryWriter(log_dir=args.save_path)

    # ===== Prepare Data =====
    all_areas = ["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"]
    test_areas = ["Area_5"]
    training_areas = sorted(list(set(all_areas) - set(test_areas)))
    logger.info("Training Areas: %s", training_areas)
    for line in textwrap.wrap(str(args), width=120):
        logger.info(line)

    trainset = S3DIStrain(args, areas=training_areas)
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=cfl_collate_fn(),
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn(seed),
    )
    clusterset = S3DIStrain(args, areas=training_areas)
    cluster_loader = DataLoader(
        clusterset,
        batch_size=1,
        collate_fn=cfl_collate_fn(),
        num_workers=args.cluster_workers,
        pin_memory=True,
    )

    model = Res16FPN18_lvwConv_4_2_CSattn_2(
        in_channels=args.input_dim,
        out_channels=args.primitive_num,
        conv1_kernel_size=args.conv1_kernel_size,
        config=args,
    )
    logger.info(model)
    model = model.cuda()

    small_lr_names = []
    for name, p in model.named_parameters():
        if "den" in name or "weight" in name:
            small_lr_names.append(name)
    params_small = [
        p for n, p in model.named_parameters() if ("den" in n or "weight" in n)
    ]
    params_rest = [
        p for n, p in model.named_parameters() if not ("den" in n or "weight" in n)
    ]
    if len(params_small) > 0:
        param_groups = [
            {"params": params_small, "lr": args.lr * 0.1},
            {"params": params_rest, "lr": args.lr},
        ]
        optimizer = torch.optim.SGD(
            param_groups,
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            dampening=args.dampening,
            weight_decay=args.weight_decay,
        )

    scheduler = PolyLR(optimizer, max_iter=args.max_iter[0])
    loss = torch.nn.CrossEntropyLoss(ignore_index=-1).cuda()

    last_checkpoints = []
    miou_records = []

    logger.info("#################################")
    logger.info("### Stage 1 ###")
    logger.info("#################################")
    is_Growing = False
    start_grow_epoch, last_checkpoints, miou_records = run_training_stage(
        args,
        logger,
        train_loader,
        cluster_loader,
        model,
        optimizer,
        scheduler,
        loss,
        start_epoch=0,
        max_epoch=args.max_epoch[0],
        max_iter=args.max_iter[0],
        is_Growing=is_Growing,
        last_checkpoints=last_checkpoints,
        miou_records=miou_records,
    )

    logger.info("#################################")
    logger.info("### Stage 2 ###")
    logger.info("#################################")
    is_Growing = False
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        dampening=args.dampening,
        weight_decay=args.weight_decay,
    )
    scheduler = PolyLR(optimizer, max_iter=args.max_iter[1])
    _, last_checkpoints, miou_records = run_training_stage(
        args,
        logger,
        train_loader,
        cluster_loader,
        model,
        optimizer,
        scheduler,
        loss,
        start_epoch=start_grow_epoch,
        max_epoch=args.max_epoch[1],
        max_iter=args.max_iter[1],
        is_Growing=is_Growing,
        last_checkpoints=last_checkpoints,
        miou_records=miou_records,
    )

    logger.info("Training Finished.")
    logger.info("Last Checkpoints: %s", [pair[0] for pair in last_checkpoints])
    logger.info(
        "Top/Second/Lowest mIoU checkpoints: %s", [(m[0], m[1]) for m in miou_records]
    )
    writer.close()


def cluster(
    args, logger, cluster_loader, model, epoch, start_grow_epoch=None, is_Growing=False
):
    time_start = time.time()
    cluster_loader.dataset.mode = "cluster"

    current_growsp = None
    if is_Growing:
        current_growsp = int(
            args.growsp_start
            - ((epoch - start_grow_epoch) / args.max_epoch[1])
            * (args.growsp_start - args.growsp_end)
        )
        if current_growsp < args.growsp_end:
            current_growsp = args.growsp_end
        logger.info("Epoch: {}, Superpoints Grow to {}".format(epoch, current_growsp))

    """Extract Superpoints Feature"""
    feats, labels, sp_index, context = get_sp_feature(
        args, cluster_loader, model, current_growsp
    )
    sp_feats = torch.cat(feats, dim=0)  ### will do Kmeans with geometric distance
    primitive_labels = KMeans(
        n_clusters=args.primitive_num, n_init=5, random_state=0, n_jobs=5
    ).fit_predict(sp_feats.numpy())
    sp_feats = sp_feats[:, 0 : args.feats_dim]  ### drop geometric feature

    """Compute Primitive Centers"""
    primitive_centers = torch.zeros((args.primitive_num, args.feats_dim))
    for cluster_idx in range(args.primitive_num):
        indices = primitive_labels == cluster_idx
        cluster_avg = sp_feats[indices].mean(0, keepdims=True)
        primitive_centers[cluster_idx] = cluster_avg
    primitive_centers = F.normalize(primitive_centers, dim=1)
    classifier = get_fixclassifier(
        in_channel=args.feats_dim,
        centroids_num=args.primitive_num,
        centroids=primitive_centers,
    )

    """Compute and Save Pseudo Labels"""
    all_pseudo, all_gt, all_pseudo_gt = get_pseudo(
        args, context, primitive_labels, sp_index
    )
    logger.info(
        "labelled points ratio %.2f clustering time: %.2fs",
        (all_pseudo != -1).sum() / all_pseudo.shape[0],
        time.time() - time_start,
    )

    """Check Superpoint/Primitive Acc in Training"""
    sem_num = args.semantic_class
    mask = (all_pseudo_gt != -1) & (all_gt != -1)
    histogram = np.bincount(
        sem_num * all_gt.astype(np.int32)[mask] + all_pseudo_gt.astype(np.int32)[mask],
        minlength=sem_num**2,
    ).reshape(
        sem_num, sem_num
    )  # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = "| mIoU {:5.2f} | ".format(100 * m_IoU)
    for IoU in IoUs:
        s += "{:5.2f} ".format(100 * IoU)
    logger.info("Superpoints oAcc {:.2f} IoUs".format(o_Acc) + s)
    Sp_s = s

    pseudo_class2gt = -np.ones_like(all_gt)
    for i in range(args.primitive_num):
        mask = all_pseudo == i
        pseudo_class2gt[mask] = torch.mode(torch.from_numpy(all_gt[mask])).values
    mask = (pseudo_class2gt != -1) & (all_gt != -1)
    histogram = np.bincount(
        sem_num * all_gt.astype(np.int32)[mask]
        + pseudo_class2gt.astype(np.int32)[mask],
        minlength=sem_num**2,
    ).reshape(
        sem_num, sem_num
    )  # hungarian matching
    o_Acc = histogram[range(sem_num), range(sem_num)].sum() / histogram.sum() * 100
    tp = np.diag(histogram)
    fp = np.sum(histogram, 0) - tp
    fn = np.sum(histogram, 1) - tp
    IoUs = tp / (tp + fp + fn + 1e-8)
    m_IoU = np.nanmean(IoUs)
    s = "| mIoU {:5.2f} | ".format(100 * m_IoU)
    for IoU in IoUs:
        s += "{:5.2f} ".format(100 * IoU)
    logger.info("Primitives oAcc {:.2f} IoUs".format(o_Acc) + s)
    P_s = s

    return classifier.cuda(), Sp_s, P_s


def train(train_loader, logger, model, optimizer, loss, epoch, scheduler, classifier):
    train_loader.dataset.mode = "train"
    model.train()
    loss_display = 0
    time_curr = time.time()
    for batch_idx, data in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx + 1

        (
            coords,
            features,
            normals,
            labels,
            inverse_map,
            pseudo_labels,
            inds,
            region,
            index,
        ) = data

        in_field = ME.TensorField(features, coords, device=0)
        feats = model(in_field)

        feats = feats[inds.long()]
        feats = F.normalize(feats, dim=-1)
        #
        pseudo_labels_comp = pseudo_labels.long().cuda()
        logits = F.linear(F.normalize(feats), F.normalize(classifier.weight))
        loss_sem = loss(logits * 3, pseudo_labels_comp).mean()

        loss_display += loss_sem.item()
        optimizer.zero_grad()
        loss_sem.backward()
        optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()
        torch.cuda.synchronize(torch.device("cuda"))

        if (batch_idx + 1) % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            logger.info(
                "Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.10f}, lr: {:.3e}, Elapsed time: {:.4f}s({} iters)".format(
                    epoch,
                    (batch_idx + 1),
                    len(train_loader),
                    100.0 * (batch_idx + 1) / len(train_loader),
                    iteration,
                    loss_display,
                    scheduler.get_lr()[0],
                    time_used,
                    args.log_interval,
                )
            )
            time_curr = time.time()
            loss_temp = loss_display
            loss_display = 0

    return loss_temp, epoch


from torch.optim.lr_scheduler import LambdaLR


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter=30000, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(
            optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step
        )


def worker_init_fn(seed):
    return lambda x: np.random.seed(seed + x)


def set_logger(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
    )
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)

    return logger


def set_seed(seed):
    """
    Unfortunately, backward() of [interpolate] functional seems to be never deterministic.

    Below are related threads:
    https://github.com/pytorch/pytorch/issues/7068
    https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi
    """
    # Use random seed.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    args = parse_args()

    """Setup logger"""
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    logger = set_logger(os.path.join(args.save_path, "train.log"))

    """Random Seed"""
    seed = args.seed
    set_seed(seed)

    main(args, logger)
