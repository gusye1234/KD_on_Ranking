
# ----------------------------------------------------------------------------
# from Procedure
def BPR_train_DNS_batch(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    global item_count
    Recmodel: PairWiseModel = recommend_model
    Recmodel.train()
    if item_count is None:
        item_count = torch.zeros(dataset.m_items)
    bpr: utils.BPRLoss = loss_class
    # S,sam_time = UniformSample_DNS_deter(allusers, dataset, world.DNS_K)
    S, negItems,NEG_scores, sam_time = UniformSample_DNS_batch(dataset, Recmodel, world.DNS_K)

    print(f"DNS[pre-sample][{sam_time[0]:.1f}={sam_time[1]:.2f}+{sam_time[2]:.2f}]")
    users = S[:, 0].long()
    posItems = S[:, 1].long()
    negItems = negItems.long()
    negScores = NEG_scores.float()
    print(negItems.shape, negScores.shape)
    users, posItems, negItems, negScores = utils.TO(users, posItems, negItems, negScores)
    users, posItems, negItems, negScores = utils.shuffle(users, posItems, negItems, negScores)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    DNS_time = time()
    DNS_time1 = 0.
    DNS_time2 = 0.
    aver_loss = 0.
    for (batch_i,
         (batch_users,
          batch_pos,
          batch_neg,
          batch_scores)) in enumerate(utils.minibatch(users,
                                                   posItems,
                                                   negItems,
                                                   negScores,
                                                   batch_size=world.config['bpr_batch_size'])):
        # batch_neg, sam_time = DNS_sampling_neg(batch_users, batch_neg, dataset, Recmodel)
        batch_neg, sam_time = DNS_sampling_batch(batch_neg, batch_scores)
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg)
        item_count[batch_neg] += 1
        DNS_time1 += sam_time[0]
        DNS_time2 += sam_time[2]
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    print(f"DNS[sampling][{time()-DNS_time:.1f}={DNS_time1:.2f}+{DNS_time2:.2f}]")
    np.savetxt(os.path.join(world.CODE_PATH, f"counts/count_{world.dataset}_{world.DNS_K}.txt"),item_count.numpy())
    aver_loss = aver_loss / total_batch
    return f"[BPR[aver loss{aver_loss:.3e}]"  