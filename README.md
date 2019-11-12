# Domain Generalization Using a Mixture of Multiple Latent Domains
![model](https://user-images.githubusercontent.com/22876486/68654944-64933100-0572-11ea-8cd0-2ff148ca1843.png)
This is the pytorch implementation of the AAAI 2020 poster paper "Domain Generalization Using a Mixture of Multiple Latent Domains".

## Requirements
- A Python install version 3.6
- A PyTorch and torchvision installation version 0.4.1 and 0.2.1, respectively. [pytorch.org](https://pytorch.org/)
- The caffe model we used for [AlexNet](https://drive.google.com/file/d/1wUJTH1Joq2KAgrUDeKJghP1Wf7Q9w4z-/view?usp=sharing)
- [PACS dataset](http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017)
- Install python requirements
```
pip install -r requirements.txt
```

## Training and Testing
You can train the model using the following command.
```
cd script
bash general.sh
```
If you want to train the model without domain generalization (Deep All), you can also use the following command.
```
cd script
bash deepall.sh
```

You can set the correct parameter.
- --data-root: the dataset folder path
- --save-root: the folder path for saving the results
- --gpu: the gpu id to run experiments

## Citation
If you use this code, please cite the following paper:

Toshihiko Matsuura and Tatsuya Harada. Domain Generalization Using a Mixture of Multiple Latent Domains. In AAAI, 2020.
```
@InProceedings{dg_mmld,
  title={Domain Generalization Using a Mixture of Multiple Latent Domains},
  author={Toshihiko Matsuura and Tatsuya Harada},
  booktitle={AAAI},
  year={2020},
  }