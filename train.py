import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# write our own functionality for the get_model
from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from utils.optimizer import load_optimizer, save_optimizer
from data.dataset import Dataset

from utils.logger import logger

# from utils.evaluation import evaluate # I created a monster
# from model import FastSpeech2Loss

from model.loss import LossHandler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    print("Prepare training ...")


    #### C
    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    #### C


    ## Prepare model and optimizer (Change checkpoint path (ckpt_path) in train.yaml), this is where the data is loaded and saved at.
    # This implementation currently names saves based on step reached (which makes sense if you only have 1 model, but all models will save at the same steps)
    # It currently loads from the step given as restore arg "args.restore_step".
    model, optimizer = get_model(args, configs, device, train=True)

    # model = nn.DataParallel(model) # USE IT IF WE WANT TO RUN ON MULTIPLE GRAPHIC CARDS

    # I guess we can do this.
    num_param = get_param_num(model)
    print("Number of FastSpeech2 Parameters:", num_param)


    ## Initialize Loss Handler
    loss_handler = LossHandler(preprocess_config['preprocessing'])

    ## Load vocoder
    vocoder = get_vocoder(model_config, device)



    # change load_path if one is wanted
    train_logger = logger(load_path = None)
    validation_logger = logger(load_path = None)

        # Remove?
        # Init logger
        # for p in train_config["path"].values():
        #     os.makedirs(p, exist_ok=True)
        # train_log_path = os.path.join(train_config["path"]["log_path"], "train")
        # val_log_path = os.path.join(train_config["path"]["log_path"], "val")
        # os.makedirs(train_log_path, exist_ok=True)
        # os.makedirs(val_log_path, exist_ok=True)
        # train_logger = SummaryWriter(train_log_path)
        # val_logger = SummaryWriter(val_log_path)


    #### C
    ## Training
    step = args.restore_step + 1
    epoch = 1
    grad_accumu_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    #### C

    ## TDQM (% bar) initialization
    if args.local:
        outer_bar = tqdm(total=total_step, desc="Training", position=0)
        outer_bar.n = args.restore_step
        outer_bar.update()

    ## Training loop
    while True:
        
        if args.local:
            # adds tdqm bar if we are running it locally
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
   
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)

                """
                id = The ID of the data point in the dataset
                
                raw_text = Text, which is NOT phoneme embedded
                
                speakers = speaker embedding, if training the model in multi speaker mode.
                
                texts = the embedded texts/sequences, on the form [B, 𝕃], where 𝕃 is the longest of all the sequences amongst the batch. Sequences not of length 𝕃 have been 0 padded as the 0 character represents the sound of nothing and is always removed if present in the text. Its current form is _

                text_lens = the original sequence lengths of the texts in the batches before they were 0 padded, this is used to generate the masks we will call the 'sequence_masks'.

                max_text_len = the max of text_lens

                mels = the target mel spectrograms for each text, which have also been 0 padded in the same manner as the text. So they have dimensionality [B, 𝕄], where 𝕄 is the longest mel sequence.    

                mel_lens = the original mel spectrogram lengths of the mel spectrograms in the batches, before they were 0 padded. 
                This is used to generate what we will call the 'frame_masks'.

                max_mel_len = max of mel_lens

                durations = the durations which have been padded such that they have the dimension [B, 𝕃] in the case of a phoneme preprocessing scheme, otherwise they would have a length of [B, 𝕄] (to match the frames? not 100% sure of this). How this will be handled is that we pass the padded texts in, but only duration extend in the slots, where we have original sequence, then we pad so it matches the max_mel_len at the end such that everything ends up having the dimensionality of [B, 𝕄, E].

                pitches = -||-

                energies = -||- 
                
                """
                ids, raw_texts, speakers, texts, text_lens, max_text_len,\
                mels, mel_lens, max_mel_len, pitches, energies, durations = batch
                # Forward
                # TODO: add explicit variable names for batch unpacking (*(batch[2:]))
                output = model(
                    speakers, 
                    texts, text_lens, max_text_len, 
                    mel_lens, max_mel_len,
                    pitches, energies, durations
                )

                mel_spectrogram_postnet, mel_spectrogram, log_duration, pitch, energy, sequence_masks, frame_masks, text_lens, mel_lens = output

                # Calculate Losses
                losses = loss_handler.get_losses(
                    mels, mel_spectrogram, mel_spectrogram_postnet,
                    durations, log_duration,
                    pitches, pitch,
                    energies, energy,
                    sequence_masks, frame_masks,
                    )
                total_loss = sum(losses)
                print(f"Total loss size: {total_loss}")

                ## Backward pass with
                total_loss = total_loss / grad_accumu_step # just set lower learning rate??
                total_loss.backward()
                
                ## clipping to avoid sudden model changes (gradient explotions)
                # nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                # ## Update params
                # optimizer.step()
                # optimizer.zero_grad()

                ## Update every grad_acc_step step
                if step % grad_accumu_step == 0:
                    ## Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    ## Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                # Rewrite this section to log and produce graphs as we want
                    # loss graphs
                if step % log_step == 0:
                    losses = [l.item() for l in losses] # check format of the tensor losses (can we jsut do .to_list()???)


                    train_logger.log_data(losses) # TODO ensure that losses is a 1d list (i assume it is from the above implementation)

                    outer_bar.write(f"Total loss of train step {step}:\n{total_loss}")

                    # message1 = "Step {}/{}, ".format(step, total_step)
                    # message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}".format(
                    #     *losses
                    # )

                    # with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    #     f.write(message1 + message2 + "\n")

                    # log(train_logger, step, losses=losses)

                
                ## synth 1 sample every synth step
                # Keep so we can see how it changes (like one per 6 hours or something)
                # TODO look further into this code (i kept everything as it was assuming it would work)
                if step % synth_step == 0:
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )


                # Write a loss loop for validation data 
                if step % val_step == 0:
                    model.eval()

                    # Get dataset # TODO use corret path to validation data
                    dataset = Dataset("val.txt", preprocess_config, train_config, sort=False, drop_last=False)
                    batch_size = train_config["optimizer"]["batch_size"]
                    loader = DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=dataset.collate_fn,)

                    # loss
                    loss_sums = [0 for _ in range(6)]
                    for batchs in loader:
                        for batch in batchs:
                            batch = to_device(batch, device)
                            with torch.no_grad():
                                # Forward
                                output = model(*(batch[2:])) # TODO Use same unpacking as in above

                                # Cal Loss
                                losses = loss_handler.get_losses(batch, output)

                                for i in range(len(losses)):
                                    loss_sums[i] += losses[i].item() * len(batch[0])

                    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]

                    validation_logger.log_data(loss_means)

                    # Remove?? (andreas), replace with model(data) -> loss(out,target) -> log
                    # message = evaluate(model, step, configs, val_logger, vocoder)
                    # with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    #     f.write(message + "\n")

                    outer_bar.write(f"Validation step{step/val_step} resulted in mean loss' of:\n{loss_means}")

                    model.train()


                # Test and understand how this works
                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )
                    quit()

                step += 1

                if args.local:
                    outer_bar.update(1)

            if args.local:  
                inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )

    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )

    parser.add_argument(
        "--local", type=bool, required=True, help="Specify whether you are training locally or on a server."
    )
    
    args = parser.parse_args()
    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
