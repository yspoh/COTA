from models import *
import torch
print(torch.__version__)
import torch.nn as nn
import os
from utils import *
# from info_nce import InfoNCE
import torch.optim as optim
from torch.optim import lr_scheduler
# New: imports for Ray Tune
from functools import partial
import tempfile
from pathlib import Path
import ray
from ray import tune
from ray.tune import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune import JupyterNotebookReporter

sigmoid = nn.Sigmoid()
# info = InfoNCE()

def test(hardRecommender, ttest_uid, ttest_iid, ttest_rates, overlap_tgt2src, device='cpu'):
    current_idx = 0
    concatenated = torch.tensor([], dtype=torch.float32, device=device)
    hardRecommender.eval()
    with torch.no_grad():
        #testing
        while True:
            batch_uid, batch_iid, batch_rates, current_idx = sample_mini_batch_sequential(
                ttest_uid, ttest_iid, ttest_rates, batch_size=BATCH_SIZE, current_idx=current_idx)
            if batch_uid is None:
                break  # reached the end
            source_buid = []
            for key in batch_uid.tolist():
                source_buid.append(overlap_tgt2src[key])
            # Source domain: all users
            sfinal_user, _, tfinal_item, _, _, _ = hardRecommender(source_buid, batch_uid, batch_iid)
            predicted_ratings = sigmoid(torch.sum(sfinal_user * tfinal_item, dim=1))
            # predicted_ratings = torch.sum(sfinal_user * tfinal_item, dim=1)
            concatenated = torch.cat([concatenated, predicted_ratings])
        # rating calculation
        concatenated = (concatenated * RATES_MULTIPLY) + RATES_ADD
        concatenated = concatenated.reshape(len(ttest_rates))

        # rating check
        # print(concatenated[1300:1320])
        # print(ttest_rates[1300:1320])
        # concatenatedp = (concatenated - RATES_ADD) / RATES_MULTIPLY
        # ttest_ratesp = (ttest_rates - RATES_ADD) / RATES_MULTIPLY
        # print(concatenatedp[1300:1320])
        # print(ttest_ratesp[1300:1320])
        # negative_count = torch.sum((concatenatedp < 0.2).float()).item()
        # print("张量中负数的数量:", negative_count)

    return float(mae(concatenated, ttest_rates).detach()), float(rmse(concatenated, ttest_rates).detach()),\
          float(r2(concatenated, ttest_rates).detach())


def train(config, pt_path):
    device = config["device"]
    usepmap = config["usepmap"]
    # otweight = config["otweight"]
    cluster_size = config["cluster_size"]
    overlap_tgt2src = config["overlap_tgt2src"]
    pre_s_weight_save = config["pre_s_weight_save"]
    pre_t_weight_save = config["pre_t_weight_save"]
    weight_name = config["weight_name"]
    newmae, newrmse, newr2 = 0, 0, 0
    wd_loss = torch.zeros(1, device=device)

    ttrain_uid, ttrain_iid, ttrain_rates, ttest_uid, ttest_iid, ttest_rates, n_user, n_item, tn_user, tn_item\
    = load_main_pt(pt_path, device)

    # Create a tensor where the index is the Target ID and the value is the Source ID
    tgt2src_tensor = torch.zeros(max(overlap_tgt2src.keys()) + 1, dtype=torch.long).to(device)
    for t_id, s_id in overlap_tgt2src.items():
        tgt2src_tensor[t_id] = s_id

    # pretrain
    lightgcn_source = MFEncoder(n_user, n_item, EMBBEDDING_DIM)
    lightgcn_source = lightgcn_source.to(device)
    lightgcn_source.load_state_dict(torch.load(pre_s_weight_save, map_location=device, weights_only=True))

    lightgcn_target = MFEncoder(tn_user, tn_item, EMBBEDDING_DIM)
    lightgcn_target = lightgcn_target.to(device)
    lightgcn_target.load_state_dict(torch.load(pre_t_weight_save, map_location=device, weights_only=True))

    # recommender
    # Train
    hardRecommender = ClusterOTRecommender(lightgcn_source, lightgcn_target, cluster_size, LAMBDA_E, MAXITER,
                                            config["num_expert"], config["tau"], config["usepmap"], device)
    hardRecommender = hardRecommender.to(device)

    # Optimizer
    optimizer_pre = optim.Adam(
        filter(lambda p: p.requires_grad, hardRecommender.parameters()),
        lr=config["lr"],
        weight_decay=WEIGHT_DECAY
    )
    scheduler_pre = lr_scheduler.StepLR(optimizer_pre, step_size=10, gamma=0.1) 

    # if config["otlr"]>0 and config["cluster_size"]>0:
    #     hardRecommender.train()
    #     for iter in range(0, ITERATIONS):
    #         _, wd_loss = hardRecommender(None, None, None, True)
    #         optimizer_ot.zero_grad()
    #         wd_loss.backward()
    #         optimizer_ot.step()

    for iter in range(0, ITERATIONS):
        hardRecommender.train()
        running_loss = 0.0
        epoch_steps = 0

        # alternate train OT per iter
        # _, wd_loss = hardRecommender(None, None, None, True)
        # if config["otlr"]>0 and config["cluster_size"]>0:
        #     optimizer_ot.zero_grad()
        #     wd_loss.backward()
        #     optimizer_ot.step()

        for batch_uid, batch_iid, batch_rates in mini_batch_iterator(ttrain_uid, ttrain_iid, ttrain_rates, batch_size=BATCH_SIZE):

            user_indices, item_indices, true_ratings\
                = batch_uid.to(device), batch_iid.to(device), batch_rates.to(device)

            # get source ids
            source_buid = tgt2src_tensor[user_indices]
            # source_buid = []
            # for key in user_indices.tolist():
            #     source_buid.append(overlap_tgt2src[key])

            sfinal_user, tfinal_user, tfinal_item, moe_loss, _, wd_loss = hardRecommender(source_buid, user_indices, item_indices)

            predicted_ratings = sigmoid(torch.sum(sfinal_user * tfinal_item, dim=1))
            # predicted_ratings = torch.sum(sfinal_user * tfinal_item, dim=1)

            # rating calculation
            true_ratings = (true_ratings - RATES_ADD) / RATES_MULTIPLY
            mse_loss = MSELOSS(predicted_ratings, true_ratings)

            train_loss = mse_loss + 0.01 * moe_loss + wd_loss * config["wdweight"]

            optimizer_pre.zero_grad()
            train_loss.backward()
            optimizer_pre.step()

            running_loss += train_loss.item()
            epoch_steps += 1

        scheduler_pre.step()

        # Check if loss is NaN
        # if np.isnan(train_loss.item()) or np.isnan(wd_loss.item()):
        #     print(f"NaN detected at iteration {iter}. Skipping trial.")
            # Report the NaN so the scheduler knows this config was bad
            # tune.report({"train_loss": train_loss.detach().cpu().numpy(), "wd_loss": float(wd_loss.detach().cpu().numpy()), "done": True})
            # return
        
        # Save checkpoint and report metrics
        checkpoint_data = {
            "epoch": iter,
            weight_name: hardRecommender.state_dict(),
            "optimizer_pre_state_dict": optimizer_pre.state_dict(),
        }

        if (iter+1) % ITERS_PER_EVAL == 0:
            newmae, newrmse, newr2 = test(hardRecommender, ttest_uid, ttest_iid, ttest_rates, overlap_tgt2src, device)
                
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint = Checkpoint.from_directory(checkpoint_dir)
            tune.report(
                {"train_loss": running_loss/epoch_steps, "iter": iter, "R2": round(newr2, 4),
                    "wd_loss": wd_loss.item(), "MAE": round(newmae, 4), "RMSE": round(newrmse, 4)},checkpoint=checkpoint)
            
            
def main(source_domain, target_domain, ratio, usepmap, num_expert, taulist,
 lrlist, wdlist, infolist, clusterlist, device, gpus_per_trial=0.25, cpus_per_trial=0.5):
    pre_s_weight_save = mf_weight[f"{source_domain}_100"]
    pre_t_weight_save = mf_weight[f"{source_domain}_{target_domain}_{ratio}"]
    weight_save = main_weight[f"{source_domain}_{target_domain}_{ratio}"]
    pt_path = os.path.join(train_data[f"{source_domain}_{target_domain}_save"], f"{ratio}train.pt")
    # target->source dict
    overlap_tgt2src = overlap(os.path.join(overlap_save, f"{target_domain}-{source_domain}-overlap.txt"))
    print("Starting hyperparameter tuning.")
    ray.init(include_dashboard=False)
    config = {
        "lr": tune.grid_search(lrlist),
        "wdweight": tune.grid_search(wdlist),
        "num_expert": tune.grid_search(num_expert),
        "tau": tune.grid_search(taulist),
        "cluster_size": tune.grid_search(clusterlist),
        "infoweight": tune.choice(infolist),
        "usepmap": usepmap,
        "pre_s_weight_save": pre_s_weight_save,
        "pre_t_weight_save": pre_t_weight_save,
        "overlap_tgt2src": overlap_tgt2src,
        "weight_name": "main_",
        "device": device,
    }
    scheduler = ASHAScheduler(
            max_t=ITERATIONS,
            grace_period=ITERATIONS,
            reduction_factor=2,
        )
    reporter = JupyterNotebookReporter(
        max_progress_rows=100,
        parameter_columns=["lr", "num_expert", "cluster_size", "tau"],
        # metric_columns=["train_loss", "wd_loss", "moe_loss"],
        metric_columns=["train_loss", "MAE", "RMSE", "iter"],
        )
    tuner = tune.Tuner(
            tune.with_resources(
                partial(train, pt_path=pt_path),
                resources={"cpu": cpus_per_trial, "gpu": gpus_per_trial}
            ),
            run_config=tune.RunConfig(
                progress_reporter=reporter,
                checkpoint_config=tune.CheckpointConfig(
                    num_to_keep=1,      # Only keep the latest checkpoint
                    checkpoint_score_attribute="train_loss", # Metric to determine "best"
                    checkpoint_score_order="min"
                )
            ),
            tune_config=tune.TuneConfig(
                metric="train_loss",
                mode="min",
                scheduler=scheduler,
                num_samples=1,
            ),
            param_space=config,
        )
    results = tuner.fit()
    best_result = results.get_best_result("train_loss", "min")
    print(f"Best trial config: {best_result.config}")
    print(f"Best trial final train loss: {best_result.metrics['train_loss']}")
    print(f"Best trial final wd loss: {best_result.metrics['wd_loss']}")

    # save all hyperparameter & parameter of model
    # best_checkpoint = best_result.checkpoint
    # with best_checkpoint.as_directory() as checkpoint_dir:
    #     checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
    #     best_checkpoint_data = torch.load(checkpoint_path)
        # torch.save(best_checkpoint, weight_save + "checkpoint_" + config["weight_name"])
        # torch.save(best_checkpoint_data[config["weight_name"]], weight_save + config["weight_name"])
        # test_acc = test_accuracy(best_trained_model, device, data_dir)
        # print(f"Best trial test set accuracy: {test_acc}")