import torch.utils.data.dataloader
import torchvision.models
import torchvision.transforms as transforms
import pickle
import argparse
from tqdm import tqdm
from PIL import Image
from metrics import MDD_loss
from dataset import *
from afenet import *




# -----------------------------args------------------------------------- #
parser = argparse.ArgumentParser(description='AFENet', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--net', type=str, default='AFENet',
                    help='Named a new network')
parser.add_argument('--dataset', type=str, default='MT-HTCD',
                    help='Options: [GZ OSCD CDD LEVIR WHU MT-HTCD MT-Wuhan]')
parser.add_argument('--optimizer', type=str, default='MultiStepLR',
                    help='Options: [LambdaLR MultiStepLR SGD]')
parser.add_argument('--batch_size', type=int, default= 4,
                    help='Batch_size')
parser.add_argument('--epoch_num', type=int, default= 1600,
                    help='epoch_num')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume_path')
args = parser.parse_args()
# ---------------------------------------------------------------------- #


# ----------------------------got args---------------------------------- #
Net = args.net
Dataset = args.dataset
Optimizer = args.optimizer
Batch_size = args.batch_size
# ---------------------------------------------------------------------- #


# ----------------------------Dataset------------------------------------- #
transforms_set = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-90, 90), expand=False)])
transforms_set_1 = transforms.Compose([
    transforms.ToTensor()])
transforms_result = transforms.ToPILImage()
if Dataset == 'GZ':
    train_data = GZ_Dataset(move='train', transform=transforms_set_1)
    test_data = GZ_Dataset(move='test', transform=transforms_set_1)
elif Dataset == 'OSCD':
    train_data = MT_OSCDDataset(move='train', transform=transforms_set_1)
    test_data = MT_OSCDDataset(move='test', transform=transforms_set_1)
elif Dataset == 'CDD':
    train_data = CDD_Dataset(move='train', transform=transforms_set_1)
    test_data = CDD_Dataset(move='test', transform=transforms_set_1)
elif Dataset == 'LEVIR':
    train_data = LEVIR_Dataset(move='train', transform=transforms_set_1)
    test_data = LEVIR_Dataset(move='test', transform=transforms_set_1)
elif Dataset == 'WHU':
    train_data = WHU_Dataset(move='train', transform=transforms_set_1)
    test_data = WHU_Dataset(move='test', transform=transforms_set_1)
elif Dataset == 'MT-HTCD':
    # data = MT_HTCDDataset_1(transform=transforms_set_1)
    # torch.manual_seed(0)
    # train_size = int(0.8 * len(data))
    # test_size = len(data) - train_size
    # train_data, test_data = torch.utils.data.random_split(data,[train_size,test_size])
    # train_data = HTCD_CSV(move='train', transform=transforms_set_1)
    # test_data = HTCD_CSV(move='test', transform=transforms_set_1)
    with open('/home/yan/Documents/CD_Code/Work1/HTCD_train_256.pickle','rb') as file:
        train_data = pickle.load(file)
    file.close()
    with open('/home/yan/Documents/CD_Code/Work1/HTCD_test_256.pickle', 'rb') as file:
        test_data = pickle.load(file)
    file.close()
elif Dataset == 'MT-Wuhan':
    train_data = MT_WuHanDataset(move='train', transform=transforms_set_1)
    test_data = MT_WuHanDataset(move='test', transform=transforms_set_1)
# ------------------------------------------------------------------------ #


# -------------------------------Model------------------------------------ #
G_model = AFENet_L(3,2)
D_model = AdversarialNetwork_L()
if args.resume is True:
    checkpoint_G = torch.load('xxx.pth')
    checkpoint_D = torch.load('xxx.pth')
    G_model.load_state_dict(checkpoint_G)
    D_model.load_state_dict(checkpoint_D)
    print('-------resume success------')
# ------------------------------------------------------------------------ #


# -------------------------------Loss------------------------------------- #
criterion_CE = nn.CrossEntropyLoss()
loss_weights = nn.Parameter(torch.ones(3).cuda())
# ------------------------------------------------------------------------ #


# ------------------------------Optimizer--------------------------------- #
if Optimizer == 'LambdaLR':
    class LambdaLR():
        def __init__(self, n_epochs, offset, decay_start_epoch):
            assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
            self.n_epochs = n_epochs
            self.offset = offset
            self.decay_start_epoch = decay_start_epoch
        def step(self, epoch):
            return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)
    parameter_list = G_model.get_parameters() + D_model.get_parameters()
    optimizer = torch.optim.Adam(parameter_list, lr=0.001, betas=(0.5, 0.999))
    optimizer.add_param_group({'params': loss_weights})
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LambdaLR(400, 0, 100).step)
elif Optimizer == 'MultiStepLR':
    parameter_list = G_model.get_parameters() + D_model.get_parameters()
    milestone = range(1100,1200,10)
    optimizer = torch.optim.Adam(parameter_list, lr=(1e-3), weight_decay=(1e-4))
    # optimizer.add_param_group({'params': D_model.parameters()})
    # optimizer.add_param_group({'params': loss_weights})
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone, gamma=0.8)
elif Optimizer == 'SGD':
    # parameter_list = G_model.get_parameters() + D_model.get_parameters()
    optimizer = torch.optim.Adam(G_model.parameters(), lr=(1e-3), weight_decay=1e-4)
    optimizer.add_param_group({'params': D_model.parameters()})
    # optimizer.add_param_group({'params': loss_weights})
# ------------------------------------------------------------------------ #


train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                           batch_size=Batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=Batch_size,
                                          shuffle=False)


def confusion_matrix(true_value, output_data):
    true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum = 0, 0, 0, 0
    output_data = torch.argmax(output_data, dim=1)
    batch_size = true_value.shape[0]
    image_size = true_value.shape[1]
    for i in range(batch_size):
        union = torch.clamp(true_value[i] + output_data[i], 0, 1)
        intersection = true_value[i] * output_data[i]
        true_positive = int(intersection.sum())
        true_negative = image_size ** 2 - int(union.sum())
        false_positive = int((output_data[i] - intersection).sum())
        false_negative = int((true_value[i] - intersection).sum())
        true_positive_sum += true_positive
        true_negative_sum += true_negative
        false_positive_sum += false_positive
        false_negative_sum += false_negative
    return true_positive_sum, true_negative_sum, false_positive_sum, false_negative_sum


def save_visual_result(output_data, img_sequence, is_label=False):
    if is_label:
        output_data = torch.heaviside(torch.squeeze(output_data), torch.tensor([0], dtype=torch.float32, device='cuda'))
    else:
        output_data = torch.argmax(output_data,dim=1)*255
        output_data = torch.squeeze(output_data).to(torch.uint8)
    output_data = output_data.cpu().clone()
    dims = len(output_data.shape)
    if dims > 2:
        batch_size = output_data.shape[0]
        for i in range(batch_size):
            image = transforms.ToPILImage()(output_data[i])
            img_sequence.append(image)
    else:
        image = transforms_result(output_data)
        img_sequence.append(image)
    return img_sequence


def evaluate(tp, tn, fp, fn):
    oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU = 0, 0, 0, 0, 0, 0, 0, 0, 0
    tp, tn, fp, fn = float(tp), float(tn), float(fp), float(fn)
    oa = (tp + tn) / (tp + tn + fp + fn)
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    if (recall + precision) > 0:
        f1 = 2 * ((precision * recall) / (precision + recall))
    if (tn + fp) > 0:
        false_alarm = fp / (tn + fp)
    if (tp + fn) > 0:
        missing_alarm = fn / (tp + fn)
    CIOU = tp / (tp + fp + fn)
    UCIOU = tn / (tn + fp + fn)
    MIOU = (CIOU + UCIOU) / 2
    return oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU


# -----------------------------train-------------------------------------- #
def train(train_loader_arg, model_G, model_D, the_epoch):
    model_G.cuda()
    model_G.train()
    model_D.cuda()
    model_D.train()
    with tqdm(total=len(train_loader_arg), desc='Train Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(train_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            out_cd, feats = model_G(img_1, img_2)
            out_cd_sm = nn.Softmax(dim=1)(out_cd)
            labels = torch.cat((label, label), dim=0)
            outputs = torch.cat((out_cd_sm, out_cd_sm), dim=0)
            label = label.squeeze(1).to(torch.long)
            entropy = Entropy(outputs)
            # MADA
            # afe_loss = 0
            # normalized_weights = torch.softmax(loss_weights, dim=0)
            # normalized_weights = [0.1,0.4,0.4]
            # for i in range(1,4):
            #     features = torch.cat((feats[0][i], feats[1][i]), dim=0)
            #     loss_ada = CADA([features, outputs], model_D, entropy=entropy, coeff=calc_coeff(batch_idx), random_layer=None)
            #     loss_dcm = MDD_loss(torch.cat((feats[0][i], feats[1][i]), dim=0), labels)
            #     loss_afe = loss_ada + 0.5*loss_dcm
            #     afe_loss += normalized_weights[i-1] * loss_afe
            features = torch.cat((feats[0][3], feats[1][3]), dim=0)
            loss_ada = CADA([features, outputs], model_D, entropy=entropy, coeff=calc_coeff(batch_idx), random_layer=None)
            # loss_dcm = MDD_loss(features, labels)
            # afe_loss = loss_ada + 0.5* loss_dcm
            cd_loss = criterion_CE(out_cd,label)
            total_loss = cd_loss + loss_ada
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t.set_postfix({'lr': '%.5f' % optimizer.param_groups[0]['lr'],
                           'loss': '%.4f' % total_loss.detach().cpu().data
                           })
            t.update(1)
    # scheduler_arg.step()


# -----------------------------test--------------------------------------- #
def test(test_loader_arg, model_arg, the_epoch):
    images = []
    images_label = []
    test_loss = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU = 0, 0, 0, 0, 0, 0, 0, 0, 0
    model_arg.cuda()
    model_arg.eval()
    with tqdm(total=len(test_loader_arg), desc='Test Epoch #{}'.format(the_epoch + 1)) as t:
        for batch_idx, (img_1, img_2, label) in tqdm(enumerate(test_loader_arg)):
            img_1, img_2, label = img_1.cuda(), img_2.cuda(), label.cuda()
            out_cd,_ = model_arg(img_1, img_2)
            label = label.squeeze(1).to(torch.long)
            tp_tmp, tn_tmp, fp_tmp, fn_tmp = confusion_matrix(label, out_cd)
            # images_label = save_visual_result(label.to(torch.float32), images_label, is_label=True)
            # images = save_visual_result(out_cd, images, is_label=False)
            tp += tp_tmp
            tn += tn_tmp
            fp += fp_tmp
            fn += fn_tmp
            if batch_idx > 10:
                oa, recall, precision, f1, false_alarm, missing_alarm, CIOU, UCIOU, MIOU = evaluate(tp, tn, fp, fn)
            t.set_postfix({'loss': '%.4f' % (test_loss / (batch_idx + 1)),
                           'acc': oa,
                           'f1': '%.4f' % f1,
                           'recall': '%.4f' % recall,
                           'precision': '%.4f' % precision,
                           'false alarm': '%.4f' % false_alarm,
                           'missing alarm': '%.4f' % missing_alarm,
                           'IoU': '%.4f' % CIOU,
                           'nIoU': '%.4f' % UCIOU,
                           'mIoU': '%.4f' % MIOU})
            t.update(1)

    if (the_epoch + 1) >= 1:
        f1_sequence.append(f1)
        f = open(Net + '_Work1_' + Dataset + '.txt', 'a')
        f.write("---------------------------------------------------\n")
        f.write("\"epoch\":\"" + "{}\"\n".format(the_epoch + 1))
        f.write("\"oa\":\"" + "{}\"\n".format(oa))
        f.write("\"f1\":\"" + "{}\"\n".format(f1))
        f.write("\"recall\":\"" + "{}\"\n".format(recall))
        f.write("\"precision\":\"" + "{}\"\n".format(precision))
        f.write("\"false alarm\":\"" + "{}\"\n".format(false_alarm))
        f.write("\"missing alarm\":\"" + "{}\"\n".format(missing_alarm))
        f.write("\"IoU\":\"" + "{}\"\n".format(CIOU))
        f.write("\"nIoU\":\"" + "{}\"\n".format(UCIOU))
        f.write("\"mIoU\":\"" + "{}\"\n".format(MIOU))
        f.write("\"best epoch\":\"" + "{}\"\n".format(f1_sequence.index(max(f1_sequence)) + 1))
        f.write('\n')
        f.write("---------------------------------------------------\n")
        f.close()
        print('max_f1:' + str(max(f1_sequence)) + ' epoch:' + str(f1_sequence.index(max(f1_sequence)) + 1) + '\n')
        if f1 == max(f1_sequence) and f1>0.5:
            torch.save(G_model.state_dict(), './save/best_G_{}_{}_{}.pth'.format(Dataset, Net, f1))
            torch.save(D_model.state_dict(), './save/best_D_{}_{}_{}.pth'.format(Dataset, Net, f1))
            # for i in range(len(images)):
            #     result_label = images_label[i]
            #     result_image = images[i]
            #     #result
            #     if not os.path.isdir('vision/' + Net + '/result/' + Dataset):
            #         os.makedirs('vision/' + Net + '/result/' + Dataset)
            #     Image.Image.save(result_image, 'vision/' + Net + '/result/' + Dataset + '/{}.png'.format(i))
            #     #label
            #     if not os.path.isdir('vision/' + Net + '/label/' + Dataset):
            #         os.makedirs('vision/' + Net + '/label/' + Dataset)
            #     Image.Image.save(result_label, 'vision/' + Net + '/label/' + Dataset + '/{}.png'.format(i))


# -----------------------------main--------------------------------------- #
f1_sequence = []
for epoch in range(args.epoch_num):
    train(train_loader_arg=train_loader,
          model_G = G_model,
          model_D = D_model,
          # scheduler_arg=scheduler,
          the_epoch=epoch)
    if epoch >= 0:
        test(test_loader_arg=test_loader,
             model_arg=G_model,
             the_epoch=epoch)
