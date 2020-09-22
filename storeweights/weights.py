import os
import torch


def show(model_name):

  if os.path.isdir(model_name):
    file_names = os.listdir(model_name)
    file_names.sort()

    if len(file_names)!=0:
      return file_names
    else:
      print('No checkpoint exists.')
  else:
    print('Not available')


def save(model_name,model,optimizer=None,extra_info={}):
  
  path = os.path.join(model_name,model_name)

  if not os.path.isdir(model_name):
    os.mkdir(model_name)
    version = 0

  else:
    file_names = os.listdir(model_name)
    file_names.sort()
    
    if len(file_names)==0:
      version = 0
    else:
      version = int(file_names[-1].split('_')[1].split('.')[0]) + 1


  file_path = os.path.join(path+f'_{version}.tar')
  
  print(f"Saving Checkpoint: {model_name}_{version}.tar")
  torch.save({'model_state_dict': model.state_dict(),
              'optimizer_state_dict': None if optimizer is None else optimizer.state_dict(),
              'extra_info':extra_info}, file_path)
  

def load(model_name,model,optimizer=None,version=None,return_extra_info=False):

  path = os.path.join(model_name,model_name)
  
  if os.path.isdir(model_name):
    file_names = os.listdir(model_name)
    file_names.sort()
    if version==None:
      version = int(file_names[-1].split('_')[1].split('.')[0])
  
  file_path = os.path.join(path+f'_{version}.tar')
    
  if os.path.isfile(file_path):
    print(f"Loading Checkpoint: {model_name}_{version}.tar")
    checkpoint = torch.load(file_path) 
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if return_extra_info:
      return checkpoint['extra_info']

  else :
    print('No checkpoint exists.')

def remove(model_name,version=None):

  path = os.path.join(model_name,model_name)
  
  if os.path.isdir(model_name):
    file_names = os.listdir(model_name)
    file_names.sort()
    if len(file_names)!=0:
      if version==None:
        print(f"Deleting {file_names[-1]}.")
        os.remove(os.path.join(model_name,file_names[-1]))

      elif os.path.isfile(os.path.join(path+f'_{version}.tar')):
        print(f"Deleting {model_name}_{version}.tar")
        os.remove(os.path.join(path+f'_{version}.tar'))

      else:
        print(f"{model_name}_{version}.tar dosen't exists.")
    else:
      print('No checkpoint exists.')

