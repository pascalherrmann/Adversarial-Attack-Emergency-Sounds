## Load Pretrained CRNN Model

```
from crnn import AudioCRNN # this might change depending on import mechanism
import torch

res_path = '/nfs/students/summer-term-2020/project-4/data/models/crnn.pth' # Path of trained model
checkpoint = torch.load(res_path)

model = AudioCRNN(checkpoint['classes'], checkpoint['config'])
model.load_state_dict(checkpoint['state_dict'])
```
