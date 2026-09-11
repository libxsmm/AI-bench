# ruff: noqa: E731

import cuda.tile as ct
from cuda.tile._backend import cpu
import torch
import torch.nn as nn

ct.set_backend("cpu")
ConstInt = ct.Constant[int]


@ct.kernel
def _depthwise_conv2d(x, weight, output, C: ConstInt, H: ConstInt, W: ConstInt, OH: ConstInt, OW: ConstInt, KH: ConstInt, KW: ConstInt, STRIDE_H: ConstInt, STRIDE_W: ConstInt, PAD_H: ConstInt, PAD_W: ConstInt, DIL_H: ConstInt, DIL_W: ConstInt, BLOCK_SIZE: ConstInt):
    nc=ct.bid(1); pid=ct.bid(0); channel=nc%C; offsets=pid*BLOCK_SIZE+ct.arange(BLOCK_SIZE,dtype=torch.int32); valid0=offsets<OH*OW; oh=offsets//OW; ow=offsets%OW; acc=ct.full((BLOCK_SIZE,),0.0,dtype=ct.float32); xm=x.get_raw_memory(); wm=weight.get_raw_memory(); max_x=C*H*W*(nc//C+1)-1
    for kh in range(KH):
        for kw in range(KW):
            ih=oh*STRIDE_H+kh*DIL_H-PAD_H; iw=ow*STRIDE_W+kw*DIL_W-PAD_W; valid=valid0&(ih>=0)&(ih<H)&(iw>=0)&(iw<W); indices=( (nc*H+ih)*W+iw); safe=ct.minimum(ct.maximum(indices,0),max_x); xv=xm.load_offset(safe,mask=safe>=0).astype(ct.float32); wi=channel*KH*KW+kh*KW+kw; wv=wm.load_offset(wi+ct.arange(1,dtype=torch.int32),mask=wi>=0).astype(ct.float32); acc+=xv*wv*valid.astype(ct.float32)
    oi=nc*OH*OW+offsets; so=ct.minimum(ct.maximum(oi,0),(nc//C+1)*C*OH*OW-1); output.get_raw_memory().store_offset(so,ct.astype(acc,output.dtype),mask=valid0)


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size_h, kernel_size_w, stride_h=1, stride_w=1, padding_h=0, padding_w=0, dilation_h=1, dilation_w=1, groups=1, bias=False):
        super().__init__(); self.conv2d=nn.Conv2d(in_channels,in_channels,(kernel_size_h,kernel_size_w),stride=(stride_h,stride_w),padding=(padding_h,padding_w),dilation=(dilation_h,dilation_w),groups=in_channels,bias=bias)

    def forward(self,x):
        x=x.contiguous(); B,C,H,W=x.shape; KH,KW=self.conv2d.kernel_size; SH,SW=self.conv2d.stride; PH,PW=self.conv2d.padding; DH,DW=self.conv2d.dilation; OH=(H+2*PH-DH*(KH-1)-1)//SH+1; OW=(W+2*PW-DW*(KW-1)-1)//SW+1; weight=self.conv2d.weight.squeeze(1).contiguous().to(x.dtype); output=torch.empty((B,C,OH,OW),device=x.device,dtype=x.dtype)
        with cpu.compile_options({"assume_in_bounds":False}): ct.launch(None,(ct.cdiv(OH*OW,32),B*C),_depthwise_conv2d,(x,weight,output,C,H,W,OH,OW,KH,KW,SH,SW,PH,PW,DH,DW,32))
        return output