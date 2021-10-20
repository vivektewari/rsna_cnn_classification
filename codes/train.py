from dataLoaders import *
from catalyst.dl import SupervisedRunner, CallbackOrder, Callback, CheckpointCallback
from config import *
from funcs import get_dict_from_class,count_parameters
from models import FeatureExtractor,modelling_3d
from losses import BCELoss
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from catalyst import dl
#from callbacks import MetricsCallback
from sklearn.model_selection import StratifiedKFold
import torch
def train(model_param,model_,data_loader_param,data_loader,loss_func,callbacks=None,pretrained=None):
    randSeed=23
    data_load = data_loader(**get_dict_from_class(data_loader_param))
    criterion = loss_func
    model = model_(**get_dict_from_class(model_param))
    count_parameters(model)
    # model = FCLayered(**get_dict_from_class(model_param,model))
    if pretrained is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(pretrained, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
        except:
            model.load_state_dict(checkpoint)
        model.eval()
    optimizer = optim.SGD(model.parameters(), lr=lr)


    train = data_load.data
    train["fold"] = -1
    temp=train['patient_id'].unique()
    temp_dict={}
    for t in temp:
        temp_dict[t]=random.randint(0,10)
    train['fold']=train['patient_id'].apply(lambda x:temp_dict[x])




    # # check the proportion
    #fold_proportion = pd.pivot_table(train, columns="fold", values="label", aggfunc=len)

    use_fold = 0

    train_file = train.query("fold != @use_fold")
    val_file = train.query("fold == @use_fold")

    print("[fold {}] train: {}, val: {}".format(use_fold, len(train_file['patient_id']), len(val_file['patient_id'])))

    loaders = {
        "train": DataLoader(data_loader( data_frame=train_file,**get_dict_from_class(data_loader_param)),
                            batch_size=2,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False),
        "valid": DataLoader(data_loader(data_frame=val_file, **get_dict_from_class(data_loader_param)),
                            batch_size=2,
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            drop_last=False)
    }

    callbacks = callbacks
    runner = SupervisedRunner(

        output_key="logits",
        input_key="image_pixels",
        target_key="targets")
    # scheduler=scheduler,

    runner.train(
        model=model,
        criterion=criterion,
        loaders=loaders,
        optimizer=optimizer,

        num_epochs=epoch,
        verbose=True,
        logdir=f"fold0",
        callbacks=callbacks,
    )

    # main_metric = "epoch_f1",
    # minimize_metric = False

if __name__ == "__main__":
    from callbacks import *

    callbacks = [MetricsCallback(input_key="targets", output_key="logits",
                        directory=saveDirectory, model_name='rsna'),dl.AUCCallback(
          input_key="logits", target_key="targets")]
    train(model_param,model,data_loader_param,data_loader,loss_func,callbacks,pretrained=pre_trained_model)