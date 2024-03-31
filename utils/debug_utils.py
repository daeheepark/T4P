import os
import os.path as osp
from pathlib import Path
import shutil

def backup_modules(conf, cur_file, output_dir):
    backup_dir = osp.join(output_dir, 'backups')
    os.makedirs(backup_dir, exist_ok=True)

    shutil.copyfile(cur_file, osp.join(backup_dir, Path(cur_file).name))

    datamodule = Path(conf.datamodule._target_.replace('.', '/')).parent.with_suffix('.py')
    shutil.copyfile(datamodule, osp.join(backup_dir, datamodule.name))

    model = Path(conf.model.target._target_.replace('.', '/')).parent.with_suffix('.py')
    shutil.copyfile(model, osp.join(backup_dir, model.name))