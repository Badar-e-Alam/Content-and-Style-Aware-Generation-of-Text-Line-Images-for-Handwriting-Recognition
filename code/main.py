import torch
import tqdm, os, glob
from parameters import device, batch_size, NUM_WRITERS
from loss import CER
from collective_train import Collective_train
from data_set import CustomImageDataset
from load_data import loadData as load_data_func
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

# from models import Visual_encoder, TextEncoder_FC
from torch import optim
import numpy as np
import time

# from encoder_vgg import Encoder
show_iter_num = 500  #
OOV = True

NUM_THREAD = 2

EARLY_STOP_EPOCH = None
ITER=1
EVAL_EPOCH = 10
MODEL_SAVE_EPOCH = 20
show_iter_num = 500
LABEL_SMOOTH = True
Bi_GRU = True
VISUALIZE_TRAIN = True

lr_dis = 1 * 1e-4
lr_gen = 1 * 1e-4
lr_rec = 1 * 1e-5
lr_cla = 1 * 1e-5
OOV = True
CurriculumModelID = 0
writer = SummaryWriter(log_dir="Logs")


def all_data_loader():
    data_train, data_test = load_data_func(OOV)

    train_loader = torch.utils.data.DataLoader(
        data_train,
        collate_fn=sort_batch,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        data_test,
        collate_fn=sort_batch,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, test_loader


def sort_batch(batch):
    train_domain = list()
    train_wid = list()
    train_idx = list()
    train_img = list()
    train_img_width = list()
    train_label = list()
    img_xts = list()
    label_xts = list()
    label_xts_swap = list()
    for (
        domain,
        wid,
        idx,
        img,
        img_width,
        label,
        img_xt,
        label_xt,
        label_xt_swap,
    ) in batch:
        if wid >= NUM_WRITERS:
            print("error!")
        train_domain.append(domain)
        train_wid.append(wid)
        train_idx.append(idx)
        train_img.append(img)
        train_img_width.append(img_width)
        train_label.append(label)
        img_xts.append(img_xt)
        label_xts.append(label_xt)
        label_xts_swap.append(label_xt_swap)

    train_domain = np.array(train_domain)
    train_idx = np.array(train_idx)
    train_wid = np.array(train_wid, dtype="int64")
    train_img = np.array(train_img, dtype="float32")
    train_img_width = np.array(train_img_width, dtype="int64")
    train_label = np.array(train_label, dtype="int64")
    img_xts = np.array(img_xts, dtype="float32")
    label_xts = np.array(label_xts, dtype="int64")
    label_xts_swap = np.array(label_xts_swap, dtype="int64")

    train_wid = torch.from_numpy(train_wid)
    train_img = torch.from_numpy(train_img)
    train_img_width = torch.from_numpy(train_img_width)
    train_label = torch.from_numpy(train_label)
    img_xts = torch.from_numpy(img_xts)
    label_xts = torch.from_numpy(label_xts)
    label_xts_swap = torch.from_numpy(label_xts_swap)

    return (
        train_domain,
        train_wid,
        train_idx,
        train_img,
        train_img_width,
        train_label,
        img_xts,
        label_xts,
        label_xts_swap,
    )


def train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch):

    """
    Function Responseible for the whole train process and optimization step

        Parameters

            Train_loader: Train dataset
            model: Combained model 
            dis_opt:  discriminator optimizer
            rec_opt: Recognizer optimizer
            cla_opt: Writer optimizer
            epoch : Number of epoch
        
        Return 

        loss_dis: discriminative loss
        loss_writer:  Writer loss
        loss_rec : Recognizer Loss
        Loss_gen : Generative loss
    
    """

    model.train()
    loss_dis = list()
    loss_dis_tr = list()
    loss_writer = list()
    loss_cla_tr = list()
    loss_gen = list()
    loss_rec = list()
    loss_rec_tr = list()
    time_s = time.time()
    print("coming into the trainign")
    for itr, train_data_list in tqdm.tqdm(enumerate(train_loader)):

        gen_opt.zero_grad()

        gen_loss = model(train_data_list, epoch, "gen_update")
        gen_opt.step()

        """rec update"""
        rec_opt.zero_grad()

        reg_loss = model(train_data_list, epoch, "rec_update")
        loss_rec_tr.append(reg_loss.item())
        rec_opt.step()
        """Generator Image"""

        """dis update"""
        dis_opt.zero_grad()

        dis_loss = model(train_data_list, epoch, "dis_update")

        loss_dis_tr.append(dis_loss)

        dis_opt.step()

        """classifier update"""
        # print("going to the classifer")
        cla_opt.zero_grad()

        writer_loss = model(train_data_list, epoch, "cla_update")
        loss_cla_tr.append(writer_loss.item())
        cla_opt.step()

        print(
            f"Iteation={ITER} --Times {time.time()-time_s}-----Generator Loss {gen_loss}--Writer loss{writer_loss} Discriminator Loss{dis_loss} Recognizaer Loss{reg_loss}"
        )
        writer.add_scalar("Generator loss", gen_loss.item(), ITER)
        writer.add_scalar("discriminative loss", dis_loss.item(), ITER)
        writer.add_scalar("Writer loss ", writer_loss.item(), ITER)
        writer.add_scalar("Recognizer  loss", reg_loss.item(), ITER)
       
        loss_dis.append(dis_loss)
        loss_writer.append(writer_loss)
        loss_rec.append(reg_loss)
        loss_gen.append(gen_loss)
        

    return np.mean(loss_dis), np.mean(loss_writer), np.mean(loss_rec), np.mean(loss_gen)


def test(test_loader, epoch, modelFile_o_model):
    print("coming into the test", epoch)
    if type(modelFile_o_model) == str:
        model = Collective_train(NUM_WRITERS, show_iter_num, OOV).to(device)
        print("Loading " + modelFile_o_model)
        model.load_state_dict(torch.load(modelFile_o_model))  # load
    else:
        model = modelFile_o_model
    model.eval()
    loss_dis = list()
    loss_cla = list()
    loss_rec = list()
    cer_te = CER()
    cer_te2 = CER()
    for test_data_list in test_loader:
        l_dis, l_cla, l_rec = model(test_data_list, epoch, "eval", [cer_te, cer_te2])

        loss_dis.append(l_dis.cpu().item())
        loss_cla.append(l_cla.cpu().item())
        loss_rec.append(l_rec.cpu().item())


def main(train_loader, test_loader):

    model = Collective_train(num_writers=NUM_WRITERS, show_iter_num=show_iter_num).to(
        device
    )
    T_parameter=sum(p.numel() for p in model.parameters())
    gen=sum(p.numel() for p in model.generative.parameters())
    w_model=sum(p.numel() for p in model.writer_Model.parameters())
    rec=sum(p.numel() for p in model.recog.parameters())
    dis=sum(p.numel() for p in model.discri.parameters())

    print(f"Total Number of Parameters::{T_parameter//1000000}.Mil ,--Generative Part::{(gen/T_parameter)*100} --Discriminator Part::{(dis/T_parameter)*100} --Writer Part::{(w_model/T_parameter)*100} --Recognizer Part::{(rec/T_parameter)*100}")

    if CurriculumModelID > 0:
        model_file = "save_weights/contran-" + str(CurriculumModelID) + ".model"
        print("Loading " + model_file)
        model.load_state_dict(torch.load(model_file))  # load

    # pretrain_dict = torch.load(model_file)
    # model_dict = model.state_dict()
    # pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and not k.startswith('gen.enc_text.fc')}
    # model_dict.update(pretrain_dict)
    # model.load_state_dict(model_dict)

    dis_params = list(model.discri.parameters())
    gen_params = list(model.generative.parameters())
    rec_params = list(model.recog.parameters())
    cla_params = list(model.writer_Model.parameters())
    dis_opt = optim.Adam([p for p in dis_params if p.requires_grad], lr=lr_dis)
    gen_opt = optim.Adam([p for p in gen_params if p.requires_grad], lr=lr_gen)
    rec_opt = optim.Adam([p for p in rec_params if p.requires_grad], lr=lr_rec)
    cla_opt = optim.Adam([p for p in cla_params if p.requires_grad], lr=lr_cla)
    epochs = 50001
    min_cer = 1e5
    min_idx = 0
    min_count = 0

    for epoch in range(CurriculumModelID, epochs):
        cer = train(train_loader, model, dis_opt, gen_opt, rec_opt, cla_opt, epoch)

        if epoch % MODEL_SAVE_EPOCH == 0:
            folder_weights = "weights"
            if not os.path.exists(folder_weights):
                os.makedirs(folder_weights)
            torch.save(model.state_dict(), folder_weights + "/contran-%d.model" % epoch)

        if epoch % EVAL_EPOCH == 0:
            test(test_loader, epoch, model)

        if EARLY_STOP_EPOCH is not None:
            if min_cer > cer:
                min_cer = cer
                min_idx = epoch
                min_count = 0
                rm_old_model(min_idx)
            else:
                min_count += 1
            if min_count >= EARLY_STOP_EPOCH:
                print("Early stop at %d and the best epoch is %d" % (epoch, min_idx))
                model_url = "weights/contran-" + str(min_idx) + ".model"
                os.system("mv " + model_url + " " + model_url + ".bak")
                os.system("rm weights/contran-*.model")
                break


def rm_old_model(index):
    models = glob.glob("save_weights/*.model")
    for m in models:
        epoch = int(m.split(".")[0].split("-")[1])
        if epoch < index:
            os.system("rm save_weights/contran-" + str(epoch) + ".model")


if __name__ == "__main__":

    """
        The CustomerImageData is actual class for the data loading is called here after the data spliting data is being sent to  main function

        Parameters:
           1    TextDatasetObj: Object of data-loading class
           2    Train and Test Holds the Train and Test set 
        Returns:
            None
    """
    print("batch_size", batch_size)
    print("program is running using::", device)
    print(time.ctime())
    TextDatasetObj = CustomImageDataset()
    train_ratio = 0.8
    test_ratio = 1 - train_ratio

    # Calculate the sizes of train and test sets based on the split ratios
    train_size = int(train_ratio * len(TextDatasetObj))
    test_size = len(TextDatasetObj) - train_size

    # Split the dataset into train and test sets
    train_set, test_set = random_split(TextDatasetObj, [train_size, test_size])

    # Define batch size and number of workers for DataLoader
    num_workers = 0

    # Create DataLoader instances for train and test sets
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    #check if data is being loaded properly
    if len(train_loader) | len(test_loader) == 0:
        print("Data isn't loaded properly")
    else:
        print(f"{len(train_loader)=}     {len(test_loader)=}")

    # train_loader, test_loader = all_data_loader()

    main(
        train_loader=train_loader, test_loader=test_loader,
    )

    # decoder_net = Decorder().to(device)
    # encoder_net = Encoder().to(device)

    # trainable_parameter = sum(
    #     param.numel() for param in encoder_net.parameters() if param.requires_grad
    # )
    # trainable_parameter_decoder = sum(
    #     param.numel() for param in decoder_net.parameters() if param.requires_grad
    # )
    # print(
    #     f" Encoder parameters = {trainable_parameter/ 1e6:.2f}.Millions \n Decoder paramters ={trainable_parameter/ 1e6:.2f}. Millions"
    # )
    # for Image, Label in tqdm.tqdm(dataset):
    #     label = pad_str(Label[0])
    #     print(f"{len(label)=} {len(label[0][0])=}")
    #     Str2Index = encoding(label=label, decoder=encoder)
    #     concate = torch.stack(Str2Index, dim=0)

    #     print(f"Shape of the labels:- {concate[0].shape}")
    #     print(f"Shape of the Image:- ", Image.shape)
    #     V_out = encoder_net(Image.to(device).unsqueeze(1))
    #     plt.imshow(V_out.cpu().detach().numpy(), cmap="gray")
    #     plt.show()
    #     T_out = decoder_net.forward(concate.to(device), V_out)  #
    #     print(f"{type(T_out)} and shape {T_out.size()}")
    #     plt.imshow(T_out.cpu().detach().numpy())
    #     plt.show()
    #     break

    # train_size = int(0.8 * (len(dataset)))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, test_size]
    # )
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=True, num_workers=no_workers
    # )
    # main(
    #     train_loader=train_data_loader,
    #     test_loader=test_data_loader,
    #     num_writers=NUM_WRITERS,
    # )

    # max_str = 0
    # str_2_int = []
    # label = list()
    # global_step = len(dataset)
    # V_encoder = Visual_encoder()
    # V_encoder.to(device=device)
    # T_encoder = TextEncoder_FC()
    # T_encoder.to(device=device)
    # # for i in tqdm.tqdm(range(10)):
    # for img, label in tqdm.tqdm(dataset):
    #     label = pad_str(label)
    #     stio_ten = decoding(
    #         label=label, decoder=encoder
    #     )  # (number_example, batch_size,text_len_max)
    #     # stio_ten = encoding(label=label, encoder=encoder)
    #     concate = torch.stack(stio_ten, dim=0)
    #     for sample in range(num_example):
    #         # (batch,512,Img,img_height,img_width)
    #         # image = torch.transpose(
    #         #     torch.transpose(img[sample], 1, 3), 2, 3
    #         # )  # [8, 342, 2270, 3] [8,3,342,2270]
    #         image = img[sample].to(device=device)
    #         print("image shape", image.shape)
    #         V_out = V_encoder(image.unsqueeze(1))
    #         T_out, ful_cat = T_encoder(concate[sample].to(device=device))
    #     break

    #     # for index in range(num_example):
    #     #     # print(img[index].shape)
    #     #     # print(stio_ten[index].size())
    #     #     V_out = V_encoder(img[index])
