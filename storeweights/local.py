import os
import torch


def localsave(model_name,model,optimizer=None,):
  
  path = os.path.join(model_name,model_name)
  sym_path = os.path.join(path,'_last.tar')

  if not os.path.isdir(model_name):
    os.mkdir(model_name)
    version = 0
  else:
    version = os.readlink(sym_path).split('_')[1].split('.')[0]
    os.unlink(sym_path)

  file_path = os.path.join(path,f'_{version}.tar')
  
  
  torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': None if optimizer is None else optimizer.state_dict()}, file_path)
  
  os.symlink(path,sym_path)

def localload(model_name,model,optimizer=None):

  path = os.path.join(model_name,model_name+'_last.tar')
  checkpoint = torch.load(os.readlink(path))

  model.load_state_dict(checkpoint['model_state_dict'])
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

