import copy
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm.auto import tqdm

from utils.ddp import is_main_process


def _pretrain_tabsyn_vae(model, data_loader, device, steps: int, log_every: int = 100, *, pretrain_epoch:int=0,
                         val_loader=None, es_patience:int=5, es_min_delta:float=1e-3, eval_every:int=100, max_val_batches:int=32):
    """
    Pretraining DDP del TabSyn-VAE:
      • si passa dal forward del modello (stage='vae_pretrain') → DDP sincronizza i gradienti,
      • si dividono i passi GLOBALI tra i rank per il vero speed-up wall-clock,
      • si abilitano i gradienti SOLO sul VAE.
    """
    import torch.distributed as dist
    def _ddp_world_rank():
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size(), dist.get_rank()
        return 1, 0

    is_ddp = isinstance(model, DDP)
    module = model.module if is_ddp else model
    world, rank = _ddp_world_rank()

    # 0) Quote di passi per rank (somma == steps globali)
    q, r = divmod(int(steps), int(world))
    local_steps = q + (1 if rank < r else 0)
    if local_steps == 0:
        if is_main_process():
            print(f"[TabSyn-VAE] steps={steps}, world={world} ⇒ alcuni rank non eseguono passi.")
        return

    if val_loader is not None and int(eval_every) > 0:
        base_local = max(1, q)  # quota condivisa da tutti i rank
        eval_every_local = int(math.ceil((int(eval_every) / max(1, int(steps))) * base_local))
        eval_every_local = max(1, min(local_steps, eval_every_local))
    else:
        eval_every_local = local_steps

    # 1) Dataloader DDP-aware: shuffling diverso dal training principale
    if hasattr(data_loader, "sampler") and hasattr(data_loader.sampler, "set_epoch"):
        data_loader.sampler.set_epoch(int(pretrain_epoch))  # stesso valore su tutti i rank

    # 2) Abilita grad SOLO sul VAE
    for p in module.parameters():
        p.requires_grad = False
    for p in module.tabsyn_vae_parameters():
        p.requires_grad = True

    # 3) Ottimizzatore sui soli parametri allenabili
    trainable = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-4)
    module.tab_vae.train()

    # --- Early stopping state -------------------------------------------------
    best_val = float('inf')
    best_vae_state = None
    patience_left = int(es_patience)
    def _evaluate_val(module, val_loader, device, max_batches: int = 32):
        if val_loader is None:
            return None  # no split → niente early stop
        if hasattr(val_loader, "sampler") and hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(int(pretrain_epoch) + 1)
        was_training = module.training
        module.tab_vae.eval()
        tot_loss = 0.0
        tot_ce = 0.0
        tot_nll = 0.0
        tot_kl = 0.0
        tot_n = 0
        nb = 0
        with torch.no_grad():
            itv = iter(val_loader)
            while nb < max_batches:
                try:
                    batch = next(itv)
                except StopIteration:
                    break
                x_tab = batch["tabular"].to(device, non_blocking=True)
                loss, ce, nll, kl = module.tabsyn_pretrain_step(x_tab, advance_counter=False)
                bsz = int(x_tab.size(0))
                tot_loss += float(loss.item()) * bsz
                tot_ce += float(ce.item()) * bsz
                tot_nll += float(nll.item()) * bsz
                tot_kl += float(kl.item()) * bsz
                tot_n += bsz
                nb += 1
        device0 = device if isinstance(device, torch.device) else torch.device(device)
        vec = torch.tensor([tot_loss, tot_ce, tot_nll, tot_kl, float(tot_n)], device=device0, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(vec, op=dist.ReduceOp.SUM)
        tot_loss, tot_ce, tot_nll, tot_kl, tot_n = vec.tolist()
        if tot_n <= 0:
            mean_loss = mean_ce = mean_nll = mean_kl = float("inf")
        else:
            mean_loss = tot_loss / tot_n
            mean_ce = tot_ce / tot_n
            mean_nll = tot_nll / tot_n
            mean_kl = tot_kl / tot_n
        if was_training:
            module.tab_vae.train()
        return {"loss": mean_loss, "ce": mean_ce, "nll": mean_nll, "kl": mean_kl}

    # 4) Progress bar (rank-0)
    pbar = tqdm(total=local_steps, desc="[TabSyn-VAE pretrain]", dynamic_ncols=True,
                leave=True, mininterval=0.1, disable=not is_main_process())
    if is_main_process():
        print(f"[TabSyn-VAE] world={world} | steps(global)={steps} | per-rank≈{q} (+1 per i primi {r})", flush=True)
        if val_loader is not None and int(eval_every) > 0:
            # print(f"[TabSyn-VAE] validation every {int(eval_every)} GLOBAL steps → "
            # f"every {eval_every_local} LOCAL steps on this rank "
            # f"(local_steps={local_steps})", flush=True)
            print(f"[TabSyn-VAE] validation every {int(eval_every)} GLOBAL steps → "
                f"every {eval_every_local} LOCAL steps on this rank "
                f"(local_steps={local_steps}, base_local={base_local})", flush=True)

    it = iter(data_loader)
    for s in range(1, local_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(data_loader)
            batch = next(it)

        # ---- TRAIN STEP ---------------------------------------
        x_tab = batch["tabular"].to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss, ce, nll, kl = model(None, x_tab, stage="vae_pretrain")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()

        if is_main_process():
            pbar.update(1)
            if (s % 10 == 0) or (s == 1):
                pbar.set_postfix_str(f"CE={ce.item():.6f} KL={kl.item():.3f} NLL={nll.item():.6f} loss={loss.item():.6f}")

        if (s % log_every == 0) or (s == 1):
            if is_main_process():
                tqdm.write(f"[TabSyn-VAE pretrain] step {s}/{local_steps} | CE={ce.item():.6f} | KL={kl.item():.6f} "
                           f"| NLL={nll.item():.6f} | loss={loss.item():.6f}")

        # ---- VALIDATION & EARLY STOP ----------------------------------------
        do_eval = (val_loader is not None) and (((s % eval_every_local) == 0) or (s == local_steps))
        if do_eval:
            valm = _evaluate_val(module, val_loader, device, max_batches=max_val_batches)
            if valm is not None:
                improved = (best_val - valm["loss"]) > float(es_min_delta)
                if is_main_process():
                    tag = "↑ best" if improved else f"↔ patience={patience_left - 1 if not improved else es_patience}"
                    print(f"[TabSyn-VAE VALID] step {s}/{local_steps} | "
                          f"loss={valm['loss']:.6f} (best={best_val:.6f}) | "
                          f"CE={valm['ce']:.6f} | NLL={valm['nll']:.6f} | KL={valm['kl']:.6f}  {tag}",
                          flush = True)
                if improved:
                    best_val = valm["loss"]
                    patience_left = int(es_patience)
                    try:
                        best_vae_state = copy.deepcopy(module.tab_vae.state_dict())
                    except Exception:
                        best_vae_state = None
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if is_main_process():
                            print(
                                f"[EARLY STOP] validation loss non migliora da {es_patience} valutazioni: stop.",
                                flush = True)
                        break

    if is_main_process():
        pbar.close()

    if is_main_process():
        module = model.module if isinstance(model, DDP) else model
        print(f"[TabSyn] quick post-pretrain sanity check on a small batch", flush=True)
        with torch.no_grad():
            try:
                it_eval = iter(data_loader)
                batch = next(it_eval)
            except StopIteration:
                it_eval = iter(data_loader)
                batch = next(it_eval)
            x_tab = batch["tabular"].to(device, non_blocking=True)
            if hasattr(module, "tab_vae") and module.tab_vae is not None:
                module.tab_vae.eval()
                m = module.tab_vae.quick_metrics(x_tab)
                print(
                    f"[TabSynVAE] quick metrics: "
                    f"cat_acc={m['cat_acc'].item():.3f} | "
                    f"num_mae={m['num_mae'].item():.4f} | "
                    f"num_var_ratio={m['num_var_ratio'].item():.3f}",
                    flush = True
                )

    # ---- Ripristina i pesi "migliori" del VAE (se disponibili) -------------
    if best_vae_state is not None:
        try:
            module.tab_vae.load_state_dict(best_vae_state, strict=True)
            if is_main_process():
                print(f"[TabSyn-VAE] restored BEST validation weights (loss={best_val:.6f}).", flush=True)
        except Exception as e:
            if is_main_process():
                print(f"[TabSyn-VAE] WARN: failed to restore best VAE state: {e}", flush=True)

    # 5) RIPRISTINO FLAG requires_grad PER IL TRAINING PRINCIPALE
    for p in module.parameters():
        p.requires_grad = True
    # Ricongela il VAE (niente grad nel training del diffusore)
    module.freeze_tab_vae(requires_grad=False)
    if hasattr(module, "tab_vae") and module.tab_vae is not None:
        module.tab_vae.eval()


__all__ = ["_pretrain_tabsyn_vae"]
