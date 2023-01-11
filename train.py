import os
import time
import warnings
import multiprocessing as mp
import gc

warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from spektral.data import BatchLoader

from model import VAE, NyanEncoder, NyanDecoder, EpochCounter
from dataset import SMILESDataset

import tensorflow as tf

#strategy = tf.distribute.MirroredStrategy()
strategy = tf.distribute.experimental.CentralStorageStrategy()
print("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope():

    model = VAE(NyanEncoder(latent_dim=64), NyanDecoder(fingerprint_bits=679, regression=1613))
    model.nyanCompile()

epochs = 100
implicit_epochs = 1
loader_epochs = 2
save = "saves/ZINC-extmodel5hk-3M-v2"
batch_size = 64
forward_processes = 0
save_every = 1
run_evaluate = True
augmentation = False

def prepare_loader (queue):

    dataset = SMILESDataset(training=True, all_data=False)
    #loader = BatchLoader(dataset, batch_size=batch_size, epochs=1, mask=True, shuffle=True)

    queue["dataset"] = dataset
    #queue["loader"] = loader

queues = list()
processes = list()

manager = mp.Manager()

if not augmentation:
    forward_processes = 0

if forward_processes > 0:

    print("Preparing parallelisation... initial compute load {}.".format(forward_processes * implicit_epochs))

for i in range(epochs):

    if forward_processes > 0:

        for j in range(forward_processes - len(processes)):

            queue = manager.dict()
            process = mp.Process(target=prepare_loader, args=[queue])
            process.start()

            processes.append(process)
            queues.append(queue)

        while True:

            found_process = False

            for j in range(len(processes)):

                if not processes[j].is_alive():

                    found_process = True
                    index = j
                    break

            if found_process:
                break

            time.sleep(0.01)

        dataset = queues[index]["dataset"]

    else:

        if augmentation or i == 0:
            # Reinitiate every time to change the conformations
            dataset = SMILESDataset(training=True, all_data=False)

    loader = BatchLoader(dataset, batch_size=batch_size, epochs=loader_epochs, mask=True, shuffle=True)
    print("Running epoch {}/{}".format((i + 1) * implicit_epochs * loader_epochs, epochs * implicit_epochs * loader_epochs))

    model.fit(loader.load(), steps_per_epoch=loader.steps_per_epoch, epochs=loader_epochs, callbacks=[EpochCounter(model.epochs)])
    time.sleep(5)

    if forward_processes > 0:

        # Delete
        del processes[index]
        del queues[index]

    if (i + 1) % save_every == 0 or i == epochs - 1:

        print("Saving model...")
        model.save(save)
        print("Model saved to {}.".format(save))

    gc.collect()

# Terminate all multiprocesses
print("Training finished! Killing processes.")

for process in processes:

    process.kill()

print("Finished.")
exit()
