import sys
import os
import torch


def load_dict(resume_path, model, only_decoder=True):
    if os.path.isfile(resume_path):
        checkpoint = torch.load(resume_path)
        # model_dict = model.state_dict()
        # print(checkpoint)
        # model_dict.update(checkpoint['state_dict'])
        state_dict = {
            k[7:]: v for k, v in checkpoint['state_dict'].items()
            if not only_decoder or (only_decoder and k[7:].startswith('decoder'))
        }
        model.load_state_dict(state_dict, strict=False)
        # delete to release more space
        del checkpoint
    else:
        sys.exit("=> No checkpoint found at '{}'".format(resume_path))
    return model
