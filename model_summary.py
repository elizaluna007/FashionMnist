import torch
from torchsummary import summary
import glob
import os
from contextlib import redirect_stdout
import io

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
files = glob.glob(r'D:\360MoveData\Users\Luna\Desktop\project\FashionMNIST\src\160_adam_k=12_dn=12\latest.pth')
for file in files:
    model = torch.load(file,device)
    f = io.StringIO()
    with redirect_stdout(f):
        summary(model, (1, 28, 28))
        s = f.getvalue()
    with open(os.path.dirname(file)+'/summary.txt','wb') as fs:
        fs.write(str.encode(s))
