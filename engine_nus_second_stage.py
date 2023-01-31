import os

from tqdm import tqdm
import torch
from torch.cuda.amp import autocast

from models.rank_loss import ranking_lossT
from utils.misc import compute_AP, compute_F1, is_main_process
import utils.lr_sched as lrs

##  train function ###
def train(text_encoder, image_encoder, optimizer, dataloader, logger, scaler, args, epoch):

    if is_main_process():
        logger.info("TRAINING MODE")
        dataloader = tqdm(dataloader)

    mean_rank_loss = 0

    for i, (train_inputs, train_labels) in enumerate(dataloader):

        lrs.adjust_learning_rate(optimizer, i / len(dataloader) + epoch, args)

        with autocast(enabled=True):

            optimizer.zero_grad()

            ### remove empty label images while training ###
            temp_label = torch.clamp(train_labels,0,1)
            temp_seen_labels = temp_label.sum(1)
            temp_label = temp_label[temp_seen_labels>0]
            train_labels   = train_labels[temp_seen_labels>0]
            train_inputs   = train_inputs[temp_seen_labels>0]

            train_inputs = train_inputs.cuda()
            train_labels = train_labels.cuda()
            
            txt_feat = text_encoder("seen")

            pred_feat, dist_feat = image_encoder.encode_img(train_inputs)

            score1 = torch.topk(pred_feat @ txt_feat.t(),k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits = (score1 + score2) / 2 

            rank_loss = ranking_lossT(logits, train_labels.float())
            loss = rank_loss

            mean_rank_loss += rank_loss.item()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            
        scaler.update()

    mean_rank_loss /= len(dataloader)

    learning_rate = optimizer.param_groups[-1]['lr']

    if is_main_process():

            logger.info("------------------------------------------------------------------")
            logger.info("FINETUNING Epoch: {}/{} \tRankLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, args.epochs, mean_rank_loss, learning_rate))
            logger.info("------------------------------------------------------------------")

            torch.save(text_encoder.state_dict(), os.path.join(args.record_path, "model_{}.pth".format(epoch)))


########### TEST FUNC ###########

def test(text_encoder, image_encoder, args, dataloader, logger, len_testdataset):

    if is_main_process():
        logger.info("=======================EVALUATION MODE=======================")
        dataloader = tqdm(dataloader)

    prediction_81 = torch.empty(len_testdataset,81)
    prediction_1006 = torch.empty(len_testdataset,1006)
    lab_81 = torch.empty(len_testdataset,81)
    lab_1006 = torch.empty(len_testdataset,1006)
    
    test_batch_size = args.test_batch_size
    
    cnt = 0

    torch.set_grad_enabled(False)
        
    txt_feat = text_encoder("all")
    for features, labels_1006, labels_81, _ in dataloader:
        
        strt = cnt
        endt = min(cnt + test_batch_size, len_testdataset)
        cnt += test_batch_size
        
        with autocast():
            
            pred_feat, dist_feat = image_encoder.encode_img(features.cuda())

            score1 = torch.topk(pred_feat @ txt_feat[925:].t(),k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat[925:].t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits81 = (score1 + score2) / 2

            score1 = torch.topk(pred_feat @ txt_feat.t(),k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits = (score1 + score2) / 2

        prediction_81[strt:endt,:] = logits81
        prediction_1006[strt:endt,:] = logits
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006

    torch.set_grad_enabled(True)

    if is_main_process():
        
        logger.info(f"sample_num:{lab_81.shape}")
        logger.info("completed calculating predictions over all images")
        logits_81_5 = prediction_81.clone()
        ap_81 = compute_AP(prediction_81.cuda(), lab_81.cuda())
        F1_3_81,P_3_81,R_3_81 = compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
        F1_5_81,P_5_81,R_5_81 = compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

        logger.info('ZSL AP: %.4f',torch.mean(ap_81))
        logger.info('k=3: %.4f,%.4f,%.4f',torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81))
        logger.info('k=5: %.4f,%.4f,%.4f',torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81))

        logits_1006_5 = prediction_1006.clone()
        ap_1006 = compute_AP(prediction_1006.cuda(), lab_1006.cuda())
        F1_3_1006,P_3_1006,R_3_1006 = compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
        F1_5_1006,P_5_1006,R_5_1006 = compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

        logger.info('GZSL AP:%.4f',torch.mean(ap_1006))
        logger.info('g_k=3:%.4f,%.4f,%.4f',torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006))
        logger.info('g_k=5:%.4f,%.4f,%.4f',torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006))

def eval(text_encoder, image_encoder, args, dataloader, len_testdataset):


    print("=======================EVALUATION MODE=======================")
    dataloader = tqdm(dataloader)

    prediction_81 = torch.empty(len_testdataset,81)
    prediction_1006 = torch.empty(len_testdataset,1006)
    lab_81 = torch.empty(len_testdataset,81)
    lab_1006 = torch.empty(len_testdataset,1006)
    
    test_batch_size = args.test_batch_size
    
    cnt = 0

    torch.set_grad_enabled(False)
        
    txt_feat = text_encoder("all")
    for features, labels_1006, labels_81, _ in dataloader:
        
        strt = cnt
        endt = min(cnt + test_batch_size, len_testdataset)
        cnt += test_batch_size
        
        with autocast():
            
            pred_feat, dist_feat = image_encoder.encode_img(features.cuda())

            score1 = torch.topk(pred_feat @ txt_feat[925:].t(),k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat[925:].t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits81 = (score1 + score2) / 2

            score1 = torch.topk(pred_feat @ txt_feat.t(),k=image_encoder.topk, dim=1)[0].mean(dim=1)
            score2 = dist_feat @ txt_feat.t()
            score1 = score1 / score1.norm(dim=-1, keepdim=True)
            score2 = score2 / score2.norm(dim=-1, keepdim=True)
            logits = (score1 + score2) / 2

        prediction_81[strt:endt,:] = logits81
        prediction_1006[strt:endt,:] = logits
        lab_81[strt:endt,:] = labels_81
        lab_1006[strt:endt,:] = labels_1006

    torch.set_grad_enabled(True)
    import pdb; pdb.set_trace()

    print("completed calculating predictions over all images")
    logits_81_5 = prediction_81.clone()
    ap_81 = compute_AP(prediction_81.cuda(), lab_81.cuda())
    F1_3_81,P_3_81,R_3_81 = compute_F1(prediction_81.cuda(), lab_81.cuda(), 'overall', k_val=3)
    F1_5_81,P_5_81,R_5_81 = compute_F1(logits_81_5.cuda(), lab_81.cuda(), 'overall', k_val=5)

    print('ZSL AP: {:.4f}'.format(torch.mean(ap_81)))
    print('k=3: {:.4f},{:.4f},{:.4f}'.format(torch.mean(F1_3_81),torch.mean(P_3_81),torch.mean(R_3_81)))
    print('k=5: {:.4f},{:.4f},{:.4f}'.format(torch.mean(F1_5_81),torch.mean(P_5_81),torch.mean(R_5_81)))

    logits_1006_5 = prediction_1006.clone()
    ap_1006 = compute_AP(prediction_1006.cuda(), lab_1006.cuda())
    F1_3_1006,P_3_1006,R_3_1006 = compute_F1(prediction_1006.cuda(), lab_1006.cuda(), 'overall', k_val=3)
    F1_5_1006,P_5_1006,R_5_1006 = compute_F1(logits_1006_5.cuda(), lab_1006.cuda(), 'overall', k_val=5)

    print('GZSL AP: {:.4f}'.format(torch.mean(ap_1006)))
    print('g_k=3: {:.4f},{:.4f},{:.4f}'.format(torch.mean(F1_3_1006), torch.mean(P_3_1006), torch.mean(R_3_1006)))
    print('g_k=5: {:.4f},{:.4f},{:.4f}'.format(torch.mean(F1_5_1006), torch.mean(P_5_1006), torch.mean(R_5_1006)))


