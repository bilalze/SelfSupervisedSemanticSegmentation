import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from math import exp,log

# class _SimpleSegmentationModel(nn.Module):
#     def __init__(self, backbone, classifier):
#         super(_SimpleSegmentationModel, self).__init__()
#         self.backbone = backbone
#         self.classifier = classifier
#         self.projection_head=
        
#     def forward(self, x):
#         input_shape = x.shape[-2:]
#         features = self.backbone(x)
#         x = self.classifier(features)
#          x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
#         return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class _SelfLearningModel2(nn.Module):
    def __init__(self, model, classifier):
        super(_SelfLearningModel2, self).__init__()
        self.model = model
        self.model.projection_head=Identity()
        self.classifier = classifier
        # self.projection_head=projection_head
        
    def forward_once(self, x):
        input_shape = x.shape[-2:]
        features = self.model(x)
        x = self.classifier(features)
        # x=self.projection_head(x)
        # x = F.interpolate(x, size=[22,22], mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x
    def forward(self, input1):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        return output1

class _SelfLearningModel(nn.Module):
    def __init__(self, backbone, classifier,projection_head):
        super(_SelfLearningModel, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.projection_head=projection_head
        
    def forward_once(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x=self.projection_head(x)
        # x = F.interpolate(x, size=[65,65], mode='bilinear', align_corners=False)
        # x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

    # def forward(self, input1, input2):
    #     # forward pass of input 1
    #     output1 = self.forward_once(input1)
    #     # forward pass of input 2
    #     output2 = self.forward_once(input2)
    #     return output1, output2

    def forward(self, input1):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        return output1
class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class WithinImageLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, temp=0.07):
        super(WithinImageLoss, self).__init__()
        self.temp = temp

    def normalization(self,x):
          return (x-x.mean()/x.std())

    # a=torch.randn(2,2,requires_grad=True)
    # b=normalization(a)
    # print(b.mean(),b.std())

    def forward(self, x0, x1,x0o,x1o):
        # euclidian distance
        # x0=self.normalization(x0)
        # x1=self.normalization(x1)
        N1=(1/4225)
        N21=torch.count_nonzero(x1o)
        N20=4225-N21
        # if N21.item() != 0:
        #   N1=1/N21.item()
        # else:
        #   N1=1
        # N20=sum([i.count(0) for i in x1o])
        # N21=sum([i.count(1) for i in x1o])
        # for p in range(len(x0o)):
        loss=0
        # print(x0o.shape)
        # print(x0.shape)
        # print(x0)
        # return
        for px,py in np.ndindex(x0o[0].shape):
            if x0o[0,px,py]==0:
                N2=1/N20
                # pass
            else:
              N2=1/N21

            tt=torch.tensordot(x1,x0[:,px,py],([0], [0]))
            d=(x0o[0,px,py]==x1o)
            tt2=torch.masked_select(tt, d)
            es = tt2/self.temp
            # print(tt2.shape)
            # print(es.shape)
            # esum=torch.sum(es)
            esum2=torch.logsumexp(tt/self.temp,[0,1])
            # esum3=torch.log(esum)
            # print(esum2)
            # sum2=0
            # counter=0
            # return
            # print('done1')
            # print(x0o.shape)
            
            # print(es.shape)
            # print(torch.masked_select(es, d).shape)
            # return
            ab=es-esum2
            sum2=torch.sum(ab)
            # print(sum2)

            # for _,qx,qy in np.ndindex(x1o.shape):
            #     if x0o[0,px,py]==x1o[0,qx,qy]:
            #         print(esum)
            #         print(es[0,qx,qy])
            #         sum2+=log(es[0,qx,qy]/esum)
            # counter+=1
            # if counter==3:
                # return
            # print(loss)
            # print(sum2)
            loss+=N2*sum2
            if torch.isnan(loss) or torch.isinf(loss):
                print(torch.max(tt))
                print(loss)
            # print(torch.cuda.memory_summary())
            # del sum2
            del es
            del d
            torch.cuda.empty_cache()

        loss=-N1*loss 
        print(loss)
        return loss

class WithinImageLoss1(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, temp=0.07):
        super(WithinImageLoss1, self).__init__()
        self.temp = temp

    def normalization(self,x):
          return (x-x.mean()/x.std())

    # a=torch.randn(2,2,requires_grad=True)
    # b=normalization(a)
    # print(b.mean(),b.std())

    def forward(self, x0, x1,x0o,x1o):
        # euclidian distance
        # x0=self.normalization(x0)
        # x1=self.normalization(x1)
        N1=(1/4225)
        N2=torch.count_nonzero(x1o)
        
        # N20=4225-N21
        # N20=sum([i.count(0) for i in x1o])
        # N21=sum([i.count(1) for i in x1o])
        # for p in range(len(x0o)):
        loss=0
        # print(x0o.shape)
        # print(x0.shape)
        # print(x0)
        # return
        for px,py in np.ndindex(x0o[0].shape):
            if x0o[0,px,py]==0:
                pass
            else:
                # N2=1/N21
              d=(1==x1o)
              tt=torch.tensordot(x1,x0[:,px,py],([0], [0]))
            # print(tt[0,0])
            # tt2=torch.dot(x1[:,0,0],x0[:,px,py])
            # print(tt2)
            # print(tt2.shape)
            # print(tt.max())
              
              tt2=torch.masked_select(tt, d)
              es = torch.exp(torch.div(tt2,self.temp))
            # print(es)
            # print(es.shape)
            # esum=torch.sum(es)
              esum2=torch.logsumexp(torch.div(tt,self.temp),[0,1])
            # esum3=torch.log(esum)
            # print(esum)
            # sum2=0
            # counter=0
            # return
            # print('done1')
            # print(x0o.shape)
            
            # print(es.shape)
            # print(torch.masked_select(es, d).shape)
            # return
              ab=torch.add(torch.log(es),-esum2)
              sum2=torch.sum(ab)
            # print(sum2)

            # for _,qx,qy in np.ndindex(x1o.shape):
            #     if x0o[0,px,py]==x1o[0,qx,qy]:
            #         print(esum)
            #         print(es[0,qx,qy])
            #         sum2+=log(es[0,qx,qy]/esum)
            # counter+=1
            # if counter==3:
                # return
            # print(loss)
            # print(sum2)
              loss+=N2*sum2
              if torch.isnan(loss) or torch.isinf(loss):
                print(loss)
            # print(torch.cuda.memory_summary())
            # del sum2
              del es
              del d
              torch.cuda.empty_cache()

        loss=-N1*loss 
        # print(loss)
        return loss

