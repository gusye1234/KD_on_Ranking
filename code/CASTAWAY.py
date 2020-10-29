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


"""
def UniformSample_DNS_batch(self, epoch, batch_score_size=512):
        # with torch.no_grad():
        #     if epoch >= self.start_epoch:
        #         self.start = True
        #     total_start = time()
        #     dataset = self.dataset
        #     dns_k = self.dns_k
        #     user_num = dataset.trainDataSize
        #     per_user_num = user_num // dataset.n_users + 1
        #     allPos = dataset.allPos
        #     S = []
        #     NEG_scores = []
        #     NEG_scores_teacher = []
        #     sample_time1 = 0.
        #     sample_time2 = 0.
        #     sample_time3 = 0.
        #     sample_time4 = 0.
        #     BinForUser = np.zeros(shape = (dataset.m_items, )).astype("int")
        #     # sample_shape = int(dns_k*1.5)+1
        #     BATCH_SCORE = None
        #     BATCH_SCORE_teacher = None
        #     now = 0
        #     NEG = np.zeros((per_user_num*dataset.n_users, dns_k))
        #     STUDENT = torch.zeros((per_user_num*dataset.n_users, dns_k))
        #     TEACHER = torch.zeros((per_user_num*dataset.n_users, dns_k))
        #     for user in range(dataset.n_users):
        #         start1 = time()
        #         if now >= batch_score_size:
        #             del BATCH_SCORE
        #             BATCH_SCORE = None
        #             BATCH_SCORE_teacher = None
        #         if BATCH_SCORE is None:
        #             left_limit = user+batch_score_size
        #             batch_list = torch.arange(user, left_limit) if left_limit <= dataset.n_users else torch.arange(user, dataset.n_users)
        #             BATCH_SCORE = self.student.getUsersRating(batch_list).cpu()
        #             BATCH_SCORE_teacher = self.teacher.getUsersRating(batch_list, t1=self.t1, t2=self.t2)
        #             now = 0
        #         sample_time1 += time()-start1

        #         start2 = time()
        #         scoreForuser = BATCH_SCORE[now]
        #         scoreForuser_teacher = BATCH_SCORE_teacher[now]
        #         # scoreForuser_teacher = BATCH_SCORE[now]
        #         now += 1
        #         posForUser = allPos[user]
        #         if len(posForUser) == 0:
        #             continue
        #         BinForUser[:] = 0
        #         BinForUser[posForUser] = 1
        #         NEGforUser = np.where(BinForUser == 0)[0]
        #         for i in range(per_user_num):
        #             start3 = time()
        #             posindex = np.random.randint(0, len(posForUser))
        #             positem = posForUser[posindex]
        #             negindex = np.random.randint(0, len(NEGforUser), size=(dns_k, ))
        #             negitems = NEGforUser[negindex]
        #             add_pair = (user, positem)
        #             # NEG_scores.append(scoreForuser[negitems])
        #             STUDENT[user*per_user_num + i, :] = scoreForuser[negitems]
        #             TEACHER[user*per_user_num + i, :] = scoreForuser_teacher[negitems]
        #             # NEG_scores_teacher.append(scoreForuser_teacher[negitems])

        #             sample_time3 += time()-start3
        #             start4 = time()
        #             S.append(add_pair)
        #             NEG[user*per_user_num + i, :] = negitems
        #             sample_time4 += time() - start4
        #         sample_time2 += time() - start2
        # # ===========================
        # if self.start:
        #     self.W *= self.beta
        # return torch.Tensor(S), torch.from_numpy(NEG), torch.stack(NEG_scores), torch.stack(NEG_scores_teacher),[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]
        return torch.Tensor(S), torch.from_numpy(NEG), STUDENT, TEACHER,[time() - total_start, sample_time1, sample_time2, sample_time3, sample_time4]
"""