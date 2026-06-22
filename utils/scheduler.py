import torch.optim.lr_scheduler as lr_scheduler

def define_scheduler(args, optimizer):
    if args.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif args.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.num_epoch / 2, gamma=0.1)
    elif args.scheduler == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif args.scheduler == 'cosine':
        # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epoch, eta_min=0)
        warmup_epochs = 2
        warmup_scheduler = lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_epochs
        )

        cosine_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.num_epoch - warmup_epochs,
            eta_min=0
        )

        scheduler = lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )


    elif args.scheduler == 'None':
        scheduler = None
    else:
        return NotImplementedError('Scheduler [{}] is not implemented'.format(args.scheduler))
    return scheduler