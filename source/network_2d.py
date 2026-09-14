import numpy as np
import sys
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn #provides all the building blocks to build the neural network
import torch.nn.functional as F
import torch.optim as optim
#from tensorflow import keras #just for downloading dataset
from torchvision import models #just for debugging
from torchvision import transforms
from torchsummary import summary #just for debugging
import time



##################################### FUNCTIONS ##############################################
#definimos nuestra neural network creando una subclase de nn.Module, inicializamos
# las layers de la neural network in __init__.
#Every nn.Module subclass implements the operations on input data in the 
#forward method

#Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#el shape es (batchsize, channels, height, width)


def get_cropdim(encoder_layer, decoder_layer):
        #crops the encoder_layer to the size of the decoder_layer so that 
    # concatenation between levels/blocks is possible.
    inputdim = encoder_layer.shape[-2]
    targetdim = decoder_layer.shape[-1]

    if (inputdim-targetdim)%2 == 0.:
       cropx = int((inputdim-targetdim)/2)
       cropy = cropx
    else:
       cropx=int((inputdim-targetdim-1)/2)
       cropy=int((inputdim-targetdim+1)/2)

    return cropx, cropy

def conv_block(in_channels, out_channels, stride, kernel_size, padding):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='circular')

def dilconv_block(in_channels, out_channels, stride, kernel_size, padding, dilation):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode='circular')

def up_conv():
    return nn.Upsample(scale_factor=2)

def get_activation():
    return nn.ReLU()

def make_pooling(stride, kernel_size, padding):
    return nn.AvgPool2d(kernel_size, stride, padding)


#multiplicado feature a feature, no de todos con todos
class MultiplicationSimple(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = conv_block(in_channels=self.in_channels + self.in_channels, out_channels=self.out_channels, stride=1, kernel_size=(1,1), padding = 0)


        # input1: (B, C1, H, W)
        # input2: (B, C2, H, W)
    def forward(self, input1, input2):
        mul = input1 * input2

        # Concatenar el producto + input1
        concat = torch.cat([mul, input1], dim=1)  # (B, 2C, H, W)

        # Pasar por la conv
        outputs = self.conv(concat)

        return outputs

class multiplication(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.conv = conv_block(in_channels=self.in_channels + self.in_channels*self.in_channels, out_channels=self.out_channels, stride=1, kernel_size=(1,1), padding = 0)

    def forward(self, input1, input2):

        multi = torch.einsum("bchw,bdhw->bcdhw", input1, input2)  # Genera (B, C1, C2, H, W)
        multi = multi.view(input1.shape[0], -1, input1.shape[2], input1.shape[3])  # (B, C1*C2, H, W)

        multi = torch.cat([multi, input1], dim=1)

        # Aplicamos la convolución
        outputs = self.conv(multi)

        return outputs


#class multiplication(nn.Module):
#    def __init__(self, in_channels, out_channels):
#        super().__init__()
#        self.conv = conv_block(
#            in_channels=in_channels + in_channels * in_channels, out_channels=out_channels, stride=1, kernel_size=1, padding=0)

#    def forward(self, input1, input2):
        """
        input1: tensor (B, C1, H, W)  -> canal lineal
        input2: tensor (B, C2, H, W)  -> canal no lineal
        """
#        B, C1, H, W = input1.shape
#        _, C2, _, _ = input2.shape

        
        # Expandimos para multiplicar todos los canales entre sí:
        # (B, C1, 1, H, W) * (B, 1, C2, H, W) -> (B, C1, C2, H, W)
        #mult = input1.unsqueeze(2) * input2.unsqueeze(1)
        
        # Aplanamos la combinación de canales: (B, C1*C2, H, W)
        #mult = mult.view(B, C1 * C2, H, W)

#        channels_multi = []

#        for i in range(C1):
#            for j in range(C2):
#                channels_multi.append(input1[:, i:i+1, :, :] * input2[:, j:j+1, :, :])

#        channels_multi.extend([input1[:, i:i+1, :, :] for i in range(C1)])
#        multilayer = torch.cat(channels_multi, dim=1)
        # Concatenamos también los canales originales de input1
        #concat = torch.cat([mult, input1], dim=1)
        
        # Aplicamos la convolución 1x1
#        out = self.conv(multilayer)
#        return out


#class multiplication_old(nn.Module):
#    def __init__(self, in_channels2, out_channels):
#        super().__init__()
#        self.out_channels = out_channels
#        self.in_channels = in_channels2
#        self.conv = conv_block(in_channels=self.in_channels, out_channels=self.out_channels, stride=1, kernel_size=(1,1), padding=0)

#    def forward(self, input1, input2):
#        channelnr_1 = input1.shape[-3]
#        channelnr_2 = input2.shape[-3]
        
#        channels_1 = []
#        for i in range(channelnr_1):
#            chan = transforms.Lambda(lambda x: x[:,i:i+1,:,:])(input1)
#            channels_1.append(chan)    
#        channels_2 = []
#        for i in range(channelnr_2):
#            chan = transforms.Lambda(lambda x: x[:,i:i+1,:,:])(input2)
#            channels_2.append(chan)
            
#        channels_multi = []
#        for chan1 in channels_1:
#            for chan2 in channels_2:
#                multi = torch.multiply(chan1,chan2)
#                channels_multi.append(multi)
                
#        for chan1 in channels_1:
#            channels_multi.append(chan1)

        #no se si este len va a funcionar     
#        if (len(channels_multi)>1):
#            multilayer = torch.cat(channels_multi, dim=1)    
#        else:
#            multilayer = channels_multi[0]  
            
#        in_channels = multilayer.shape[1]
#        outputs = self.conv(multilayer)    
#        return outputs

# dilatacion, residual block (with dilation y batch_norm), attention

class dilconv_block_nonlineal(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dilconv_block = dilconv_block(self.in_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=4, dilation=4)
        self.act = get_activation()

    def forward(self, x): 

        out = self.dilconv_block(x)
        out = self.act(out)

        return out

class dilconv_block_multi(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dilconv_block = dilconv_block(self.in_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=4, dilation=4)
        self.multi = multiplication(self.out_channels, self.out_channels)

    def forward(self, x_lin, x_nonlin): 

        x_lin = self.dilconv_block(x_lin)
        out = self.multi(x_lin, x_nonlin)

        return out

class ResidualDilatedBlock(nn.Module): 

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad_size = padding

        #self.reflect_pad = nn.ReflectionPad2d(self.pad_size) 

        # padding = 0 porque ya hay reflect pad del tamaño de padding
        self.conv1 = dilconv_block(self.in_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=self.pad_size, dilation=self.dil)
        self.bn1   = nn.BatchNorm2d(self.out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = dilconv_block(self.out_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=self.pad_size, dilation=self.dil)
        self.bn2   = nn.BatchNorm2d(self.out_channels)
        self.skip  = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1,1)) if self.in_channels != self.out_channels else nn.Identity()

    def forward(self, x):

        residual = self.skip(x)
        #out = self.reflect_pad(x)
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.relu(out)

        #out = self.reflect_pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        return out + residual #self.relu(out + residual)

# donwblock con residuals y dilataciones
class downblock_res_dil(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad = padding
        self.res = ResidualDilatedBlock(self.in_channels, self.out_channels, self.dil, self.pad)
        self.avg = make_pooling(stride=2, kernel_size=(3,3), padding=1)

    def forward(self, x):
        skip = self.res(x)
        down = self.avg(skip)
        return skip, down

class ResidualDilatedBlock_nonlineal(nn.Module): 

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad_size = padding

        #self.reflect_pad = nn.ReflectionPad2d(self.pad_size) 

        # padding = 0 porque ya hay reflect pad del tamaño de padding
        self.conv1 = dilconv_block(self.in_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=self.pad_size, dilation=self.dil)
        self.bn1   = nn.BatchNorm2d(self.out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = dilconv_block(self.out_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=self.pad_size, dilation=self.dil)
        self.bn2   = nn.BatchNorm2d(self.out_channels)
        self.skip  = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=(1,1)) if self.in_channels != self.out_channels else nn.Identity()

    def forward(self, x):

        residual = self.skip(x)
        #out = self.reflect_pad(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        #out = self.reflect_pad(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out + residual #self.relu(out + residual)

class downblock_res_dil_nonlineal(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad = padding
        self.res = ResidualDilatedBlock_nonlineal(self.in_channels, self.out_channels, self.dil, self.pad)
        self.avg = make_pooling(stride=2, kernel_size=(3,3), padding=1)

    def forward(self, x):
        skip = self.res(x)
        down = self.avg(skip)
        return skip, down

class downblock_res_dil_nonlineal_multi(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad = padding
        self.res = ResidualDilatedBlock_nonlineal(self.in_channels, self.out_channels, self.dil, self.pad)
        self.avg = make_pooling(stride=2, kernel_size=(3,3), padding=1)
        self.multi = MultiplicationSimple(self.out_channels, self.out_channels)

    def forward(self, x_nonlin, x_nonlin_in):
        skip = self.res(x_nonlin)
        skip = self.multi(skip, x_nonlin_in)
        down = self.avg(skip)
        return skip, down

class downblock_res_dil_multi(nn.Module):

    def __init__(self, in_channels, out_channels, dilation, padding):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dil = dilation
        self.pad = padding
        self.res = ResidualDilatedBlock(self.in_channels, self.out_channels, self.dil, self.pad)
        self.avg = make_pooling(stride=2, kernel_size=(3,3), padding=1)
        self.multi = MultiplicationSimple(self.out_channels, self.out_channels)

    def forward(self, x_lin, x_nonlin):
        skip = self.res(x_lin)
        skip = self.multi(skip, x_nonlin)
        down = self.avg(skip)
        return skip, down


# downblock con dilataciones en la convolución
class downblock_lineal_dil(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = dilconv_block(self.in_channels, self.out_channels, stride=1, kernel_size=(3,3), padding=2, dilation=2)
        self.avg = make_pooling(stride=2, kernel_size=(3,3), padding=1)
        self.act = get_activation()

    def forward(self, x_lin):

        encoder_lineal = self.conv(x_lin)
        encoder_lineal = self.avg(encoder_lineal)
    
        return encoder_lineal

#attention block

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # decoder_channels = canales que llegan al upblock (sin concatenar el skip)
        # skip_channels = canales del encoder skip
        # inter_channels = canales internos de la atención, típicamente skip_channels // 2
        # proyecta decoder y skip a un espacio común
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1)  # produce mapa de atención
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, decoder, encoder):
        # proyecciones lineales
        g1 = self.W_g(decoder)        # decoder
        x1 = self.W_x(encoder)        # skip

        # combinación + activación
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))  # mapa de atención [0,1]

        # multiplicación elemento a elemento
        return encoder * psi


class upblock_attention(nn.Module):

    def __init__(self, in_channels, skip_channels, out_channels): 
        super().__init__()
        self.decoder_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

    #5 decoders blocks que tienen 1 upsampling y 1 conv 
        self.conv = conv_block(self.decoder_channels + self.skip_channels, self.out_channels, stride=1, kernel_size=3, padding=1)
        self.up = up_conv()
        self.act = get_activation()
        self.att_gate = AttentionGate(F_g=self.decoder_channels, F_l=self.skip_channels, F_int=self.out_channels)


    def forward(self, decoder_lin, encoder_lin):

        decoder_lin = self.up(decoder_lin)
        encoder_att = self.att_gate(decoder_lin, encoder_lin)
        
        skip_lin = torch.cat((encoder_att, decoder_lin), dim=1)
        decoder_lin = self.conv(skip_lin)

        return decoder_lin



class upblock_nonlineal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = conv_block(self.in_channels, self.out_channels, stride=1, kernel_size=self.kernel_size, padding=self.padding)
        self.up = up_conv()
        self.act = get_activation()


    def forward(self, decoder_nonlin, encoder_nonlin):
        
        decoder_up = self.up(decoder_nonlin)        
        skip_nonlin = torch.cat((encoder_nonlin, decoder_up), dim=1)
        decoder_nonlin = self.conv(skip_nonlin)
        decoder_nonlin = self.act(decoder_nonlin)

        return decoder_nonlin, decoder_up


class upblock_lineal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

    #5 decoders blocks que tienen 1 upsampling y 1 conv 
        self.conv = conv_block(self.in_channels, self.out_channels, stride=1, kernel_size=self.kernel_size, padding=self.padding)
        self.up = up_conv()
        self.act = get_activation()


    def forward(self, decoder_lin, encoder_lin):
        
        decoder_lin = self.up(decoder_lin)        
        skip_lin = torch.cat((encoder_lin, decoder_lin), dim=1)
        decoder_lin = self.conv(skip_lin)

        return decoder_lin

# multiplicación ANTES de concatenacion y convolucion
class upblock_lineal_multi(nn.Module):
    def __init__(self, in_channels, in_channels2, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.in_channels = in_channels
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = conv_block(self.in_channels, self.out_channels, stride=1, kernel_size=self.kernel_size, padding=self.padding)
        self.up = up_conv()
        self.act = get_activation()
        self.multi = MultiplicationSimple(self.in_channels2, self.in_channels2)

    def forward(self, decoder_lin, encoder_lin, decoder_nonlin):
        
        decoder_lin = self.up(decoder_lin)
        decoder_lin = self.multi(decoder_lin, decoder_nonlin)
        skip_lin = torch.cat((encoder_lin, decoder_lin), dim=1)
        decoder_lin = self.conv(skip_lin)

        #decoder_lin = self.multi(decoder_lin, decoder_nonlin)

        return decoder_lin

class upblock_nonlineal_multi(nn.Module):
    def __init__(self, in_channels, in_channels2, out_channels, kernel_size=3, padding=1): 
        super().__init__()
        self.in_channels = in_channels
        self.in_channels2 = in_channels2
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = conv_block(self.in_channels, self.out_channels, stride=1, kernel_size=self.kernel_size, padding=self.padding)
        self.up = up_conv()
        self.act = get_activation()
        self.multi = MultiplicationSimple(self.in_channels2, self.in_channels2)

    def forward(self, decoder_nonlin, encoder_nonlin, decoder_nonlin_in):
        
        decoder_nonlin = self.up(decoder_nonlin)
        decoder_nonlin_up = self.multi(decoder_nonlin, decoder_nonlin_in)
        skip_nonlin = torch.cat((encoder_nonlin, decoder_nonlin_up), dim=1)
        decoder_nonlin = self.conv(skip_nonlin)
        decoder_nonlin = self.act(decoder_nonlin)

        return decoder_nonlin, decoder_nonlin_up


########################### CNN #############################################################
# lo mismo pero con activaciones en las capas, actualizar para que resiva QU y Mask
class DeepWiener_nonlineal(nn.Module):

    # mapa 512x512

    # en los channels de init en wienernet tienene que ir los shape del tensor
    # inicia con 2 y termina con 1

    def __init__(self,in_channels, out_channels, filters, kernel_size, padding):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        [self.filters0, self.filters1, self.filters2, self.filters3, self.filters4] = filters
 
        #encoder lineal (map data)

        self.down_lin0 = downblock_nonlineal(in_channels=1, out_channels=self.filters0, kernel_size = self.kernel_size, padding = self.padding)
        self.down_lin1 = downblock_nonlineal(in_channels=self.filters0, out_channels=self.filters1, kernel_size = self.kernel_size, padding = self.padding)
        self.down_lin2 = downblock_nonlineal(in_channels=self.filters1, out_channels=self.filters2, kernel_size = self.kernel_size, padding = self.padding)
        self.down_lin3 = downblock_nonlineal(in_channels=self.filters2, out_channels=self.filters3, kernel_size = self.kernel_size, padding = self.padding)
        self.down_lin4 = downblock_nonlineal(in_channels=self.filters3, out_channels=self.filters4, kernel_size = self.kernel_size, padding = self.padding)

        #decoder lineal (map data)

        self.up_lin4 = upblock_nonlineal(in_channels=(self.filters4), out_channels=self.filters4, kernel_size = self.kernel_size, padding = self.padding)
        self.up_lin3 = upblock_nonlineal(in_channels=(self.filters4+self.filters3), out_channels=self.filters3, kernel_size = self.kernel_size, padding = self.padding)
        self.up_lin2 = upblock_nonlineal(in_channels=(self.filters3+self.filters2), out_channels=self.filters2, kernel_size = self.kernel_size, padding = self.padding)
        self.up_lin1 = upblock_nonlineal(in_channels=(self.filters2+self.filters1), out_channels=self.filters1, kernel_size = self.kernel_size, padding = self.padding)
        self.up1_lin0 = up_conv()
        self.up2_lin0 = conv_block(in_channels=(self.filters1+self.filters0), out_channels=1, stride=1, kernel_size=3, padding=1)
 


    # ver como inicializar

    #initialize the weights, hay que pasar una funcion de inicializacion 
    # que inicialice los weights en todos los modulos recursivamente
        self.initialize_parameters()

    #si el modulo es alguno de esos tipos lo inicializas con el method
        #solo el conv2d tiene weights
    @staticmethod
    def weight_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.weight)

    @staticmethod
    def bias_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.bias)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_):
        for module in self.modules():
            self.weight_init(module, method_weights)
            self.bias_init(module, method_bias)


    #initial data for the tensor
    def forward(self, x):

        #el tensor que recibe es (batch_size, 1, H, W) # (batch, 1, 512, 512)
        #los dos channels son por el mapdata y maskdata

        mapdata = x[:, 0:2, :, :] # (batch, 2, 512, 512) # QObs, Uobs    
        #maskdata = transforms.Lambda(lambda x: x[:,2:3,:,:])(x)  #Mask
        #inhodata = transforms.Lambda(lambda x: x[:,2:4,:,:])(x)  #Mask and Inho

        #encoder lineal (map data)

        encoder0_lin = self.down_lin0(mapdata)
        encoder1_lin = self.down_lin1(encoder0_lin)
        encoder2_lin = self.down_lin2(encoder1_lin)
        encoder3_lin = self.down_lin3(encoder2_lin)
        encoder4_lin = self.down_lin4(encoder3_lin)

        #decoder lineal (map data)

        skip3_lin, decoder4_lin = self.up_lin4(encoder4_lin, encoder3_lin)
        skip2_lin, decoder3_lin = self.up_lin3(skip3_lin, encoder2_lin)
        skip1_lin, decoder2_lin = self.up_lin2(skip2_lin, encoder1_lin)
        skip0_lin, decoder1_lin = self.up_lin1(skip1_lin, encoder0_lin)
  
        decoder0_lin = self.up1_lin0(skip0_lin)
        decoder0_lin = self.up2_lin0(decoder0_lin)
        
        return decoder0_lin


# agregar dilatación en el encoder
class DeepWiener_dilation(nn.Module):

    # mapa 512x512

    # en los channels de init en wienernet tienene que ir los shape del tensor
    # inicia con 2 y termina con 1

    def __init__(self,in_channels, out_channels, filters):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        [self.filters0, self.filters1, self.filters2, self.filters3, self.filters4, self.filters5] = filters
 
        #encoder lineal (map data)

        self.down_lin0 = downblock_res_dil(in_channels=self.in_channels, out_channels=self.filters0, dilation=1, padding=1)
        self.down_lin1 = downblock_res_dil(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_lin2 = downblock_res_dil(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_lin3 = downblock_res_dil(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_lin4 = downblock_res_dil(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck = dilconv_block(in_channels = self.filters4, out_channels = self.filters5, stride=1, kernel_size=(3,3), padding=4, dilation=4)

        #decoder lineal (map data)

        self.up_lin4 = upblock_lineal(in_channels=(self.filters5+self.filters4), out_channels=self.filters4)
        self.up_lin3 = upblock_lineal(in_channels=(self.filters4+self.filters3), out_channels=self.filters3)
        self.up_lin2 = upblock_lineal(in_channels=(self.filters3+self.filters2), out_channels=self.filters2)
        self.up_lin1 = upblock_lineal(in_channels=(self.filters2+self.filters1), out_channels=self.filters1)
        self.up_lin0 = upblock_lineal(in_channels=(self.filters1+self.filters0), out_channels=self.out_channels)



    # ver como inicializar

    #initialize the weights, hay que pasar una funcion de inicializacion 
    # que inicialice los weights en todos los modulos recursivamente
        self.initialize_parameters()

    #si el modulo es alguno de esos tipos lo inicializas con el method
        #solo el conv2d tiene weights
    @staticmethod
    def weight_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.weight)

    @staticmethod
    def bias_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.bias)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_):
        for module in self.modules():
            self.weight_init(module, method_weights)
            self.bias_init(module, method_bias)


    #initial data for the tensor
    def forward(self, x):

        #el tensor que recibe es (batch_size, 1, H, W) # (batch, 1, 512, 512)
        #los dos channels son por el mapdata y maskdata


        mapdata = x[:, 0:2, :, :] # (batch, 2, 512, 512) #Qobs, Uobs
        #maskdata = transforms.Lambda(lambda x: x[:,2:3,:,:])(x)  #Mask
        #inhodata = transforms.Lambda(lambda x: x[:,2:4,:,:])(x)  #Mask and Inho

        #encoder lineal (map data)

        skip0_lin, down0_lin = self.down_lin0(mapdata) #512 to 256
        skip1_lin, down1_lin = self.down_lin1(down0_lin) # 256 to 128
        skip2_lin, down2_lin = self.down_lin2(down1_lin) # 128 to 64
        skip3_lin, down3_lin = self.down_lin3(down2_lin) # 64 to 32
        skip4_lin, down4_lin = self.down_lin4(down3_lin) # 32 to 16

        b1_lin = self.bottleneck(down4_lin) #16x16

        #decoder lineal (map data)

        decoder4_lin = self.up_lin4(b1_lin, skip4_lin) #32x32
        decoder3_lin = self.up_lin3(decoder4_lin, skip3_lin) #64x64
        decoder2_lin = self.up_lin2(decoder3_lin, skip2_lin) #128x128
        decoder1_lin = self.up_lin1(decoder2_lin, skip1_lin) #256x256
        decoder0_lin = self.up_lin0(decoder1_lin, skip0_lin) #512x512
        
        return decoder0_lin



# skip connections con attention (que tienen non lineal)
class DeepWiener_attention(nn.Module):

    # mapa 512x512

    # en los channels de init en wienernet tienene que ir los shape del tensor
    # inicia con 2 y termina con 1

    def __init__(self,in_channels, out_channels, filters):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        [self.filters0, self.filters1, self.filters2, self.filters3, self.filters4, self.filters5] = filters
 
        #encoder lineal (map data)

        self.down_lin0 = downblock_res_dil(in_channels=self.in_channels, out_channels=self.filters0, dilation=1, padding=1)
        self.down_lin1 = downblock_res_dil(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_lin2 = downblock_res_dil(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_lin3 = downblock_res_dil(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_lin4 = downblock_res_dil(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck = dilconv_block(in_channels = self.filters4, out_channels = self.filters5, stride=1, kernel_size=(3,3), padding=4, dilation=4)

        #decoder lineal (map data)

        self.up_lin4 = upblock_attention(in_channels=(self.filters5), skip_channels=self.filters4, out_channels=self.filters4)
        self.up_lin3 = upblock_attention(in_channels=(self.filters4), skip_channels=self.filters3, out_channels=self.filters3)
        self.up_lin2 = upblock_attention(in_channels=(self.filters3), skip_channels=self.filters2, out_channels=self.filters2)
        self.up_lin1 = upblock_attention(in_channels=(self.filters2), skip_channels=self.filters1, out_channels=self.filters1)
        self.up_lin0 = upblock_attention(in_channels=(self.filters1), skip_channels=self.filters0, out_channels=self.out_channels)
 

    # ver como inicializar

    #initialize the weights, hay que pasar una funcion de inicializacion 
    # que inicialice los weights en todos los modulos recursivamente
        self.initialize_parameters()

    #si el modulo es alguno de esos tipos lo inicializas con el method
        #solo el conv2d tiene weights
    @staticmethod
    def weight_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.weight)

    @staticmethod
    def bias_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.bias)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_):
        for module in self.modules():
            self.weight_init(module, method_weights)
            self.bias_init(module, method_bias)


    #initial data for the tensor
    def forward(self, x):

        #el tensor que recibe es (batch_size, 1, H, W) # (batch, 1, 512, 512)
        #los dos channels son por el mapdata y maskdata


        mapdata = x[:, 0:2, :, :] # (batch, 2, 512, 512) # Qobs, Uobs
        #maskdata = transforms.Lambda(lambda x: x[:,2:3,:,:])(x)  #Mask
        #inhodata = transforms.Lambda(lambda x: x[:,2:4,:,:])(x)  #Mask and Inho

        #encoder lineal (map data)
        skip0_lin, down0_lin = self.down_lin0(mapdata) #512 to 256
        skip1_lin, down1_lin = self.down_lin1(down0_lin) # 256 to 128
        skip2_lin, down2_lin = self.down_lin2(down1_lin) # 128 to 64
        skip3_lin, down3_lin = self.down_lin3(down2_lin) # 64 to 32
        skip4_lin, down4_lin = self.down_lin4(down3_lin) # 32 to 16

        b1_lin = self.bottleneck(down4_lin) #16x16

        #decoder lineal (map data)

        decoder4_lin = self.up_lin4(b1_lin, skip4_lin) #32x32
        decoder3_lin = self.up_lin3(decoder4_lin, skip3_lin) #64x64
        decoder2_lin = self.up_lin2(decoder3_lin, skip2_lin) #128x128
        decoder1_lin = self.up_lin1(decoder2_lin, skip1_lin) #256x256
        decoder0_lin = self.up_lin0(decoder1_lin, skip0_lin) #512x512
        
        return decoder0_lin


class DeepWiener_twochannels(nn.Module):

    # mapa 512x512

    # en los channels de init en wienernet tienene que ir los shape del tensor
    # inicia con 2 y termina con 1

    def __init__(self,in_channels, out_channels, filters):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        [self.filters0, self.filters1, self.filters2, self.filters3, self.filters4, self.filters5] = filters
 
        #encoder non lineal (mask data)

        self.down_nonlin0 = downblock_res_dil_nonlineal(in_channels=1, out_channels=self.filters0, dilation=1, padding=1)
        self.down_nonlin1 = downblock_res_dil_nonlineal(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_nonlin2 = downblock_res_dil_nonlineal(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_nonlin3 = downblock_res_dil_nonlineal(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_nonlin4 = downblock_res_dil_nonlineal(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck_nonlin = dilconv_block_nonlineal(in_channels = self.filters4, out_channels = self.filters5)

        #decoder non lineal (mask data)

        #up, skip connection, conv
        self.up_nonlin4 = upblock_nonlineal(in_channels=(self.filters5+self.filters4), out_channels=self.filters4)
        self.up_nonlin3 = upblock_nonlineal(in_channels=(self.filters4+self.filters3), out_channels=self.filters3)
        self.up_nonlin2 = upblock_nonlineal(in_channels=(self.filters3+self.filters2), out_channels=self.filters2)
        self.up_nonlin1 = upblock_nonlineal(in_channels=(self.filters2+self.filters1), out_channels=self.filters1)


        #encoder lineal (map data)

        # conv, multi
        self.down_lin0 = downblock_res_dil_multi(in_channels=self.in_channels, out_channels=self.filters0, dilation=1, padding=1)
        self.down_lin1 = downblock_res_dil_multi(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_lin2 = downblock_res_dil(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_lin3 = downblock_res_dil(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_lin4 = downblock_res_dil(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck = dilconv_block(in_channels = self.filters4, out_channels = self.filters5, stride=1, kernel_size=(3,3), padding=4, dilation=4)

        #decoder lineal (map data)

        # up, multiplication, skip connection, conv
        self.up_lin4 = upblock_lineal_multi(in_channels=(self.filters5+self.filters4), in_channels2=self.filters5, out_channels=self.filters4)
        self.up_lin3 = upblock_lineal_multi(in_channels=(self.filters4+self.filters3), in_channels2=self.filters4, out_channels=self.filters3)
        self.up_lin2 = upblock_lineal_multi(in_channels=(self.filters3+self.filters2), in_channels2=self.filters3, out_channels=self.filters2)
        self.up_lin1 = upblock_lineal_multi(in_channels=(self.filters2+self.filters1), in_channels2=self.filters2, out_channels=self.filters1)
        self.up_lin0 = upblock_lineal(in_channels=(self.filters1+self.filters0), out_channels=self.out_channels)


    # ver como inicializar

    #initialize the weights, hay que pasar una funcion de inicializacion 
    # que inicialice los weights en todos los modulos recursivamente
        self.initialize_parameters()

    #si el modulo es alguno de esos tipos lo inicializas con el method
        #solo el conv2d tiene weights
    @staticmethod
    def weight_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.weight)

    @staticmethod
    def bias_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.bias)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_):
        for module in self.modules():
            self.weight_init(module, method_weights)
            self.bias_init(module, method_bias)


    #initial data for the tensor
    def forward(self, x):

        #el tensor que recibe es (batch_size, 1, H, W) # (batch, 1, 512, 512)
        #los dos channels son por el mapdata y maskdata


        mapdata = x[:, 0:2, :, :] # (batch, 2, 512, 512) #Qobs, Uobs
        maskdata = x[:, 2:3, :, :] # (batch, 1, 512, 512) # mask
        #inhodata = transforms.Lambda(lambda x: x[:,2:4,:,:])(x)  #Mask and Inho

        #encoder nonlineal (mask data)

        skip0_nonlin, down0_nonlin = self.down_nonlin0(maskdata) #512 to 256
        skip1_nonlin, down1_nonlin = self.down_nonlin1(down0_nonlin) # 256 to 128
        skip2_nonlin, down2_nonlin = self.down_nonlin2(down1_nonlin) # 128 to 64
        skip3_nonlin, down3_nonlin = self.down_nonlin3(down2_nonlin) # 64 to 32
        skip4_nonlin, down4_nonlin = self.down_nonlin4(down3_nonlin) # 32 to 16

        b1_nonlin = self.bottleneck_nonlin(down4_nonlin) #16x16

        #decoder nonlineal (mask data)

        decoder4_nonlin, up4_nonlin = self.up_nonlin4(b1_nonlin, skip4_nonlin) #32x32
        decoder3_nonlin, up3_nonlin = self.up_nonlin3(decoder4_nonlin, skip3_nonlin) #64x64
        decoder2_nonlin, up2_nonlin = self.up_nonlin2(decoder3_nonlin, skip2_nonlin) #128x128
        decoder1_nonlin, up1_nonlin = self.up_nonlin1(decoder2_nonlin, skip1_nonlin) #256x256

        #encoder lineal (map data)

        skip0_lin, down0_lin = self.down_lin0(mapdata, skip0_nonlin) #512 to 256
        skip1_lin, down1_lin = self.down_lin1(down0_lin, skip1_nonlin) # 256 to 128
        skip2_lin, down2_lin = self.down_lin2(down1_lin) # 128 to 64
        skip3_lin, down3_lin = self.down_lin3(down2_lin) # 64 to 32
        skip4_lin, down4_lin = self.down_lin4(down3_lin) # 32 to 16

        b1_lin = self.bottleneck(down4_lin) #16x16

        #decoder lineal (map data)

        decoder4_lin = self.up_lin4(b1_lin, skip4_lin, up4_nonlin) #32x32
        decoder3_lin = self.up_lin3(decoder4_lin, skip3_lin, up3_nonlin) #64x64
        decoder2_lin = self.up_lin2(decoder3_lin, skip2_lin, up2_nonlin) #128x128
        decoder1_lin = self.up_lin1(decoder2_lin, skip1_lin, up1_nonlin) #256x256
        decoder0_lin = self.up_lin0(decoder1_lin, skip0_lin) #512x512
        
        return decoder0_lin

class DeepWiener_threechannels(nn.Module):

    # mapa 512x512

    # en los channels de init en wienernet tienene que ir los shape del tensor
    # inicia con 2 y termina con 1

    def __init__(self,in_channels, out_channels, filters):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        [self.filters0, self.filters1, self.filters2, self.filters3, self.filters4, self.filters5] = filters

        #enconder non lineal (inho data)

        self.down_nonlin0_in = downblock_res_dil_nonlineal(in_channels=2, out_channels=self.filters0, dilation=1, padding=1)
        self.down_nonlin1_in = downblock_res_dil_nonlineal(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_nonlin2_in = downblock_res_dil_nonlineal(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_nonlin3_in = downblock_res_dil_nonlineal(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_nonlin4_in = downblock_res_dil_nonlineal(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck_nonlin_in = dilconv_block_nonlineal(in_channels = self.filters4, out_channels = self.filters5)

        #decoder nonlineal (inho data)

        self.up_nonlin4_in = upblock_nonlineal(in_channels=(self.filters5+self.filters4), out_channels=self.filters4)
        self.up_nonlin3_in = upblock_nonlineal(in_channels=(self.filters4+self.filters3), out_channels=self.filters3)
        self.up_nonlin2_in = upblock_nonlineal(in_channels=(self.filters3+self.filters2), out_channels=self.filters2)
        self.up_nonlin1_in = upblock_nonlineal(in_channels=(self.filters2+self.filters1), out_channels=self.filters1)
 
        #encoder non lineal (mask data)

        self.down_nonlin0 = downblock_res_dil_nonlineal_multi(in_channels=1, out_channels=self.filters0, dilation=1, padding=1)
        self.down_nonlin1 = downblock_res_dil_nonlineal_multi(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_nonlin2 = downblock_res_dil_nonlineal(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_nonlin3 = downblock_res_dil_nonlineal(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_nonlin4 = downblock_res_dil_nonlineal(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck_nonlin = dilconv_block_nonlineal(in_channels = self.filters4, out_channels = self.filters5)

        #decoder non lineal (mask data)

        #up, skip connection, conv
        self.up_nonlin4 = upblock_nonlineal_multi(in_channels=(self.filters5+self.filters4), in_channels2=self.filters5, out_channels=self.filters4)
        self.up_nonlin3 = upblock_nonlineal_multi(in_channels=(self.filters4+self.filters3), in_channels2=self.filters4, out_channels=self.filters3)
        self.up_nonlin2 = upblock_nonlineal_multi(in_channels=(self.filters3+self.filters2), in_channels2=self.filters3, out_channels=self.filters2)
        self.up_nonlin1 = upblock_nonlineal_multi(in_channels=(self.filters2+self.filters1), in_channels2=self.filters2, out_channels=self.filters1)


        #encoder lineal (map data)

        # conv, multi
        self.down_lin0 = downblock_res_dil_multi(in_channels=self.in_channels, out_channels=self.filters0, dilation=1, padding=1)
        self.down_lin1 = downblock_res_dil_multi(in_channels=self.filters0, out_channels=self.filters1, dilation=1, padding=1)
        self.down_lin2 = downblock_res_dil(in_channels=self.filters1, out_channels=self.filters2, dilation=2, padding=2)
        self.down_lin3 = downblock_res_dil(in_channels=self.filters2, out_channels=self.filters3, dilation=2, padding=2)
        self.down_lin4 = downblock_res_dil(in_channels=self.filters3, out_channels=self.filters4, dilation=4, padding=4)

        self.bottleneck = dilconv_block(in_channels = self.filters4, out_channels = self.filters5, stride=1, kernel_size=(3,3), padding=4, dilation=4)

        #decoder lineal (map data)

        # up, multiplication, skip connection, conv
        self.up_lin4 = upblock_lineal_multi(in_channels=(self.filters5+self.filters4), in_channels2=self.filters5, out_channels=self.filters4)
        self.up_lin3 = upblock_lineal_multi(in_channels=(self.filters4+self.filters3), in_channels2=self.filters4, out_channels=self.filters3)
        self.up_lin2 = upblock_lineal_multi(in_channels=(self.filters3+self.filters2), in_channels2=self.filters3, out_channels=self.filters2)
        self.up_lin1 = upblock_lineal_multi(in_channels=(self.filters2+self.filters1), in_channels2=self.filters2, out_channels=self.filters1)
        self.up_lin0 = upblock_lineal(in_channels=(self.filters1+self.filters0), out_channels=self.out_channels)


    # ver como inicializar

    #initialize the weights, hay que pasar una funcion de inicializacion 
    # que inicialice los weights en todos los modulos recursivamente
        self.initialize_parameters()

    #si el modulo es alguno de esos tipos lo inicializas con el method
        #solo el conv2d tiene weights
    @staticmethod
    def weight_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.weight)

    @staticmethod
    def bias_init(module, method):
        if isinstance(module, (nn.Conv2d)):
            method(module.bias)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_):
        for module in self.modules():
            self.weight_init(module, method_weights)
            self.bias_init(module, method_bias)


    #initial data for the tensor
    def forward(self, x):

        #el tensor que recibe es (batch_size, 1, H, W) # (batch, 1, 512, 512)
        #los dos channels son por el mapdata y maskdata


        mapdata = x[:, 0:2, :, :] # (batch, 2, 512, 512) #Qobs, Uobs
        maskdata = x[:, 2:3, :, :] # (batch, 1, 512, 512) # mask
        inhodata = x[:, 2:4, :, :] # (batch, 2, 512, 512) # inho
        #inhodata = transforms.Lambda(lambda x: x[:,2:4,:,:])(x)  #Mask and Inho

        #encoder nonlineal (inho data)

        skip0_nonlin_in, down0_nonlin_in = self.down_nonlin0_in(inhodata) #512 to 256
        skip1_nonlin_in, down1_nonlin_in = self.down_nonlin1_in(down0_nonlin_in) # 256 to 128
        skip2_nonlin_in, down2_nonlin_in = self.down_nonlin2_in(down1_nonlin_in) # 128 to 64
        skip3_nonlin_in, down3_nonlin_in = self.down_nonlin3_in(down2_nonlin_in) # 64 to 32
        skip4_nonlin_in, down4_nonlin_in = self.down_nonlin4_in(down3_nonlin_in) # 32 to 16

        b1_nonlin_in = self.bottleneck_nonlin_in(down4_nonlin_in) #16x16

        #decoder nonlineal (inho data)
        
        decoder4_nonlin_in, up4_nonlin_in = self.up_nonlin4_in(b1_nonlin_in, skip4_nonlin_in) #32x32
        decoder3_nonlin_in, up3_nonlin_in = self.up_nonlin3_in(decoder4_nonlin_in, skip3_nonlin_in) #64x64
        decoder2_nonlin_in, up2_nonlin_in = self.up_nonlin2_in(decoder3_nonlin_in, skip2_nonlin_in) #128x128
        decoder1_nonlin_in, up1_nonlin_in = self.up_nonlin1_in(decoder2_nonlin_in, skip1_nonlin_in) #256x256

        #encoder nonlineal (mask data)

        skip0_nonlin, down0_nonlin = self.down_nonlin0(maskdata, skip0_nonlin_in) #512 to 256
        skip1_nonlin, down1_nonlin = self.down_nonlin1(down0_nonlin, skip1_nonlin_in) # 256 to 128
        skip2_nonlin, down2_nonlin = self.down_nonlin2(down1_nonlin) # 128 to 64
        skip3_nonlin, down3_nonlin = self.down_nonlin3(down2_nonlin) # 64 to 32
        skip4_nonlin, down4_nonlin = self.down_nonlin4(down3_nonlin) # 32 to 16

        b1_nonlin = self.bottleneck_nonlin(down4_nonlin) #16x16

        #decoder nonlineal (mask data)

        decoder4_nonlin, up4_nonlin = self.up_nonlin4(b1_nonlin, skip4_nonlin, up4_nonlin_in) #32x32
        decoder3_nonlin, up3_nonlin = self.up_nonlin3(decoder4_nonlin, skip3_nonlin, up3_nonlin_in) #64x64
        decoder2_nonlin, up2_nonlin = self.up_nonlin2(decoder3_nonlin, skip2_nonlin, up2_nonlin_in) #128x128
        decoder1_nonlin, up1_nonlin = self.up_nonlin1(decoder2_nonlin, skip1_nonlin, up1_nonlin_in) #256x256

        #encoder lineal (map data)

        skip0_lin, down0_lin = self.down_lin0(mapdata, skip0_nonlin) #512 to 256
        skip1_lin, down1_lin = self.down_lin1(down0_lin, skip1_nonlin) # 256 to 128
        skip2_lin, down2_lin = self.down_lin2(down1_lin) # 128 to 64
        skip3_lin, down3_lin = self.down_lin3(down2_lin) # 64 to 32
        skip4_lin, down4_lin = self.down_lin4(down3_lin) # 32 to 16

        b1_lin = self.bottleneck(down4_lin) #16x16

        #decoder lineal (map data)

        decoder4_lin = self.up_lin4(b1_lin, skip4_lin, up4_nonlin) #32x32
        decoder3_lin = self.up_lin3(decoder4_lin, skip3_lin, up3_nonlin) #64x64
        decoder2_lin = self.up_lin2(decoder3_lin, skip2_lin, up2_nonlin) #128x128
        decoder1_lin = self.up_lin1(decoder2_lin, skip1_lin, up1_nonlin) #256x256
        decoder0_lin = self.up_lin0(decoder1_lin, skip0_lin) #512x512
        
        return decoder0_lin



