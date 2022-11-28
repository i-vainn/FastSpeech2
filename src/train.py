import torch
import os
import wandb

from torch import nn
from tqdm import tqdm


def train(model, fastspeech_loss, optimizer, scheduler, logger, training_loader, train_config):
    tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) * train_config.batch_expand_size - current_step)

    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                tqdm_bar.update(1)
                
                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                energy_target = db["energy_target"].to(train_config.device)
                pitch_target = db["pitch_target"].to(train_config.device)
                duration_target = db["duration"].int().to(train_config.device)
                speaker_id = torch.tensor(db["speaker_id"]).int().to(train_config.device) - 1
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                (
                    mel_output,
                    duration_prediction,
                    energy_prediction,
                    pitch_prediction,
                ) = model(
                    character,
                    src_pos,
                    mel_pos=mel_pos,
                    mel_max_length=max_mel_len,
                    length_target=duration_target,
                    energy_target=energy_target,
                    pitch_target=pitch_target,
                    speaker_id=speaker_id,
                )

                # Calc Loss
                losses = fastspeech_loss.forward(
                    mel=mel_output,
                    duration_predicted=duration_prediction,
                    pitch_predicted=pitch_prediction,
                    energy_predicted=energy_prediction,
                    mel_target=mel_target,
                    duration_target=torch.log(duration_target + 1),
                    pitch_target=pitch_target,
                    energy_target=energy_target,
                )
                
                total_loss = sum(losses)

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                logger.add_scalar("total_loss", t_l)
                names = ["mel_loss", "duration_predictor_loss", "energy_predictor_loss", "pitch_predictor_loss"]
                for loss, name in zip(losses, names):
                    logger.add_scalar(name, loss.detach().cpu().numpy())
                logger.add_scalar("learning_rate", optimizer.param_groups[0]['lr'])
                logger.add_image("learning_rate", mel_output[0])

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    os.makedirs(train_config.checkpoint_path, exist_ok=True)
                    save_path = os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step)
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, save_path)
                    wandb.save(save_path)
                    print("save model at step %d ..." % current_step)
