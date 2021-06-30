import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import torch
from dataset import get_tts_dataset
import hparams as hp
from distribute import apply_gradient_allreduce
from loss_function import Tacotron2Loss
from model import Tacotron2
from logger import Tacotron2Logger
import argparse
import torch.distributed as dist
from datetime import datetime, timedelta
import time
import numpy as np
from tqdm import tqdm

from utils import is_pytorch_16plus, stream


def init_distributed(n_gpus, rank, group_name):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."
    if rank == 0:
        print("Let's use", n_gpus, "GPUs!")
        print("Initializing Distributed")
    torch.cuda.set_device(rank % torch.cuda.device_count())
    dist.init_process_group(
        backend=hp.dist_backend, init_method=hp.dist_url,
        world_size=n_gpus, rank=rank, group_name=group_name)
    if rank == 0:
        print("Done initializing distributed")


def reduce_tensor(tensor, n_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n_gpus
    return rt


def save_checkpoint(model, optimizer, learning_rate, global_step):
    k = global_step // 1000
    if is_pytorch_16plus:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'iteration': global_step},
                   os.path.join(hp.checkpoint_path, f'checkpoint_{k}k_steps.pyt'),
                   _use_new_zipfile_serialization=False)
    else:
        torch.save({'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'learning_rate': learning_rate,
                    'iteration': global_step},
                   os.path.join(hp.checkpoint_path, f'checkpoint_{k}k_steps.pyt'))


def prepare_directories_and_logger(output_directory, log_directory, rank):
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        logger = Tacotron2Logger(os.path.join(output_directory, log_directory))
    else:
        logger = None
    return logger


def train(model, optimizer, learning_rate, train_set, num_gpus, rank, current_step):
    global_step = current_step
    total_iters = len(train_set)
    epochs = hp.epochs
    epoch_offset = max(0, int(current_step / len(train_set)))

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hp.lr_decay, last_epoch=-1)
    criterion = Tacotron2Loss(total_iters)
    logger = prepare_directories_and_logger(hp.checkpoint_path, "logs", rank)

    for e in range(epoch_offset, epochs):
        if global_step > 200000: break
        start = time.time()
        clock = datetime.now()
        running_loss = 0.0

        for i, (ids, phones, mels, gates, speaker_ids, input_lengths, mel_lengths) in enumerate(train_set, 1):
            if global_step*num_gpus > 50000:
                for param_group in optimizer.state_dict()["param_groups"]:
                    param_group["lr"] = 1e-5
            elif global_step*num_gpus > 20000:
                for param_group in optimizer.state_dict()["param_groups"]:
                    param_group["lr"] = 1e-4
            else:
                for param_group in optimizer.state_dict()["param_groups"]:
                    param_group["lr"] = 1e-3

            model.zero_grad()

            phones, mels = phones.cuda(), mels.cuda()
            input_lengths = input_lengths.cuda()
            mel_lengths = mel_lengths.cuda()
            gates = gates.cuda()
            speaker_ids = speaker_ids.cuda()

            model_outputs = model(phones, mels, input_lengths, mel_lengths)
            loss = criterion(global_step, model_outputs, [mels, gates, input_lengths, mel_lengths], speaker_ids, speaker_ids)

            running_loss += reduce_tensor(loss.data, num_gpus).item() if hp.distributed_run else loss.item()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

            optimizer.step()
            k = global_step // 1000

            speed = i / (time.time() - start)
            epoch_total_times = total_iters / speed
            estimated_finished_time = clock + timedelta(seconds=epoch_total_times * (epochs - epoch_offset - e - i / total_iters))
            train_finished_time = estimated_finished_time.strftime("%m-%d %H:%M")

            if rank == 0:
                msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {running_loss/i:#.4} | {speed:#.2} steps/s | Step: {k}k | finished_time: {train_finished_time}'
                stream(f"\r{msg}")
                if k > 0 and global_step % hp.save_checkpoint_every_n_step == 0:
                    save_checkpoint(model, optimizer, learning_rate, global_step)
                logger.log_training(running_loss, grad_norm, learning_rate, global_step)

            global_step += 1

        if rank == 0:
            print(' ')

        # scheduler.step()


def create_gta_features(model, train_set, save_path):
    for (ids, phones, mels, gates, speaker_ids, input_lengths, mel_lengths) in tqdm(train_set):
        phones, mels = phones.cuda(), mels.cuda()
        input_lengths = input_lengths.cuda()
        mel_lengths = mel_lengths.cuda()
        # speaker_ids = speaker_ids.cuda()
        model.eval()
        with torch.no_grad():
            model_outputs = model(phones, mels, input_lengths, mel_lengths)
        gta = model_outputs[1].cpu().numpy()
        for j in range(len(ids)):
            np.save(os.path.join(save_path, ids[j]), gta[j][:, :mel_lengths[j]])


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_gta', action='store_true', help='Force the model to create GTA features')
    parser.add_argument("--weights_path", '-w', default=None, type=str, help='Choose the weights to restore')
    parser.add_argument("--warm_start", default=None, type=str, help='Choose the weights to warm start')
    parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
    parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
    parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
    args = parser.parse_args()


    if hp.distributed_run and not args.force_gta:
        init_distributed(args.n_gpus, rank=args.rank, group_name=args.group_name)

    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed(hp.seed)
    model = Tacotron2(hp).cuda()
    learning_rate = hp.learning_rate
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)


    if hp.distributed_run and not args.force_gta:
        model = apply_gradient_allreduce(model)

    if args.weights_path is not None:
        checkpoint_dict = torch.load(args.weights_path, map_location='cpu')
        model.load_state_dict(checkpoint_dict['state_dict'])
        current_step = checkpoint_dict['iteration']
        learning_rate = checkpoint_dict['learning_rate']
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    else:
        current_step = 0

    if args.warm_start is not None:
        print("warming start...")
        checkpoint_dict = torch.load(args.warm_start, map_location='cpu')
        model.load_state_dict(checkpoint_dict['state_dict'])

    # print model parameters
    if args.rank == 0:
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.product(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3fM' % parameters)

    if not args.force_gta:
        train_set = get_tts_dataset(args.force_gta)
        training_steps = hp.training_steps - current_step
        os.makedirs(hp.checkpoint_path, exist_ok=True)
        train(model, optimizer, learning_rate, train_set, args.n_gpus, args.rank, current_step)

        print('Training Complete.')
        exit(0)

    print('Creating Ground Truth Aligned Dataset...\n')
    assert args.weights_path is not None
    os.makedirs(hp.gta_path, exist_ok=True)
    train_set = get_tts_dataset(args.force_gta)
    create_gta_features(model, train_set, hp.gta_path)
