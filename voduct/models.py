import torch
import torch.nn as nn
from torch.nn import ReLU
import numpy as np
import time
import os
import torch.nn.functional as F
import sencoder as sen
import matplotlib.pyplot as plt
from crab.custom_modules import Sin, Transformer, TransformerBase
import crab.custom_modules as crabmods

DEVICE_DICT = {-1:"cpu", 0:"cuda:0"}

SEQ2SEQ = "seq2seq"
AUTOENCODER = "autoencoder"
DICTIONARY = "dictionary"
TRANSFORMER_TYPE = {"Transformer":SEQ2SEQ,
                    "TransAutoencoder":AUTOENCODER,
                    "Codt":AUTOENCODER,
                    "LSTMBaseline":AUTOENCODER}
CONV = "convolution"
TRANS = "transformer"
RSSM = "rssm"
COLLAPSE_TYPES = {"convolution":CONV,
                  "transformer":TRANS,
                  "conv":CONV,
                  "trans":TRANS,
                  "rssm":RSSM,
                  "rnn":RSSM}

EXPAND_TYPES = {"convolution":CONV,
                "transformer":TRANS,
                "conv":CONV,
                "trans":TRANS,
                "rssm":RSSM,
                "rnn":RSSM}

class CustomBase(TransformerBase):
    def __init__(self, **kwargs):
        """
        seq_len: int or None
            the maximum length of the sequences to be analyzed. If None,
            dec_slen and enc_slen must not be None
        enc_slen: int or None
            the length of the sequences to be encoded
        dec_slen: int or None
            the length of the sequences to be decoded
        n_vocab: int
            the number of words in the vocabulary
        emb_size: int
            the size of the embeddings
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        enc_layers: int
            the number of encoding layers
        dec_layers: int
            the number of decoding layers
        enc_mask: bool
            if true, encoder uses a mask
        class_h_size: int
            the size of the hidden layers in the classifier
        class_bnorm: bool
            if true, the classifier uses batchnorm
        class_drop_p: float
            the dropout probability for the classifier
        act_fxn: str
            the activation function to be used in the MLPs
        collapse_type: str
            the type of collapsing module for the embedding encoding
        expand_type: str
            the type of expanding module for the embedding encoding
        enc_drop_ps: float or list of floats
            the dropout probability for each encoding layer
        dec_drop_ps: float or list of floats
            the dropout probability for each decoding layer
        collapse_drop_p: float
            the dropout probability for the collapsing layer
        expand_drop_p: float
            the dropout probability for the expanding layer
        n_filts: int
            the number of filters for the partial convolutions.
            The embedding size is divided by this number to get the
            number of convoltion chunks.
        ordered_preds: bool
            if true, the decoder will mask the predicted sequence so
            that the attention modules will not see the tokens ahead
            located further along in the sequence.
        gen_decs: bool
            if true, decodings are generated individually and used
            as the inputs for later decodings. (stands for generate
            decodings). This ensures earlier attention values are
            completely unaffected by later inputs.
        init_decs: bool
            if true, an initialization decoding vector is learned as
            the initial input to the decoder.
        idx_inputs: bool
            if true, the inputs are integer (long) indexes that require
            an embedding layer. Otherwise it is assumed that the inputs
            are feature vectors that do not require an embedding layer
        """
        super().__init__(**kwargs)

######### EXPANSION AND REDUCTION MODULES

class ExpandingAttention(nn.Module):
    """
    This is a mechanism to expand a single embedding into an entire
    sequence. This is a tool to train systems to define words and
    use definitions of words as a means to understanding words.
    """
    def __init__(self, emb_size, seq_len, attn_size, n_heads,
                                                     bnorm=False,
                                                     act_fxn="Sin"):
        super().__init__()
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.attn_size = attn_size
        self.n_heads = n_heads

        modules = []
        modules.append(nn.Linear(emb_size, seq_len*emb_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(seq_len*emb_size))
        modules.append(globals()[act_fxn]())
        modules.append(nn.Linear(seq_len*emb_size,
                                 seq_len*emb_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(seq_len*emb_size))
        modules.append(globals()[act_fxn]())
        self.dec_net = nn.Sequential(*modules)

        self.multi_attn = MultiHeadAttention(emb_size,attn_size,
                                             n_heads=n_heads)
        self.norm1 = nn.LayerNorm((emb_size,))
        self.norm2 = nn.LayerNorm((emb_size,))

    def forward(self, x):
        """
        x: float tensor (B,E)
            the encoded embeddings made to represent a sequence of
            encodings
        """
        fx = self.dec_net(x) # (B,S*E)
        fx = fx.reshape(len(fx),self.seq_len,self.emb_size)
        fx = self.norm1(fx)
        attn = self.multi_attn(fx,fx,fx)
        return self.norm2(fx+attn)

class ExpandingRSSM(nn.Module):
    """
    A device to expand a single embedding into a sequence of
    embeddings using an Recurrent State-Space Model.
    """
    def __init__(self, emb_size, seq_len, h_size=None,
                                          attn_size=64,
                                          n_heads=8):
        """
        emb_size: int
            the size of the embeddings
        h_size - int
            size of belief vector h
        attn_size: int
            the size of the projected spaces in the attention layers
        n_heads: int
            the number of attention heads
        """
        super().__init__()
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.h_size = emb_size if h_size is None else h_size
        self.attn_size = attn_size
        self.n_heads = n_heads

        self.rssm = sen.models.RSSM(h_size=self.h_size,
                                    s_size=self.emb_size,
                                    emb_size=self.emb_size)
        self.multi_attn = MultiHeadAttention(emb_size=self.emb_size,
                                             attn_size=attn_size,
                                             n_heads=n_heads)

    def forward(self, encs):
        """
        encs: torch FloatTensor (B,E)
            a batch of encoded embeddings

        Returns:
            s: torch FloatTensor (B,S,E)
                the expanded sequence hopefully making up a definition
                or sentence.
        """
        # h_tup is a tuple of tensors [(B,H),(B,S),(B,S)]
        h_tup = self.rssm.init_h(batch_size=len(encs))
        context = torch.zeros_like(encs)
        seq = []
        mus = None
        sigs = None
        for i in range(self.seq_len):
            h_tup = self.rssm.state_fwd(encs,h_tup,context)
            mu,sig = h_tup[1][:,None], h_tup[2][:,None]
            if i == 0:
                mus = mu
                sigs = sig
            else:
                mus = torch.cat([mus,mu],axis=1)
                sigs = torch.cat([sigs,sig],axis=1)
            s = mus+torch.randn_like(sigs)*sigs
            if i == self.seq_len-1:
                return s
            context = self.multi_attn(s,s,s)[:,-1]

class ReductionConvolution(nn.Module):
    def __init__(self, emb_size, n_filts=8, k_sizes=[3,5,7],
                                            drop_p=0):
        """
        This convolution type serves to recursively reduce a sequence
        to a single vector of the same dimensionality as the embedding.

        emb_size: int
            the size of the embeddings
        n_filts: int
            the number of separate depthwise filters. Maximum possible
            would be the embedding size. The depthwise filters are
            shared between the emb_size/n_filts chanels that each
            depthwise filter is assigned to.
        k_sizes: list of ints or int
            each integer argued corresponds to a filter that will span
            that many words in the sequence. Each of these filters
            will have the number of output channels specified by
            n_chans
        drop_p: float
            the dropout probability
        """
        super().__init__()
        k_sizes = sorted(k_sizes)
        self.emb_size = emb_size
        self.k_sizes = k_sizes
        self.n_filts = n_filts
        self.drop_p = drop_p

        min_k = int(np.min(k_sizes))
        min_shave = min_k//2

        self.k1conv = nn.Conv1d(emb_size,emb_size,1)
        k_sizes = list(filter(lambda x: x!=1,k_sizes))
        convs = nn.ModuleList([])
        for ksize in k_sizes:
            shave = ksize//2
            padding = shave-min_shave
            conv = nn.Conv1d(emb_size, emb_size, ksize,
                                                 padding=padding,
                                                 groups=n_filts)
            convs.append(conv)
        self.convs = convs
        self.recomb = nn.Conv2d(len(k_sizes), 1, 1)
        self.nonlin = Sin()

    def forward(self,x,*args):
        bsize = len(x)
        fx = x.permute(0,2,1) # (B,E,S)
        if fx.shape[-1]%2==0:
            shape = (*fx.shape[:-1],1)
            zeros = torch.zeros(shape).to(DEVICE_DICT[fx.get_device()])
            fx = torch.cat([fx,zeros],axis=-1)
        reductions = []
        while len(fx.shape) > 2:
            fx = self.k1conv(fx)
            fxs = []
            for conv in self.convs:
                fxs.append(conv(fx)[:,None])
            fx = torch.cat(fxs, axis=1) # (B,N,E,S')
            fx = self.recomb(fx).squeeze() # (B,E,S')
            if bsize == 1:
                fx = fx[None]
            fx = self.nonlin(fx)
        return fx

class CollapsingAttention(nn.Module):
    """
    This is a mechanism to collapse an entire sequence into a single 
    embedding. This is a tool to train systems to define words and
    use definitions of words as a means to understanding words.
    """
    def __init__(self, emb_size, seq_len, attn_size, n_heads,
                                                     bnorm=False,
                                                     act_fxn="Sin"):
        super().__init__()
        self.emb_size = emb_size
        self.seq_len = seq_len
        self.attn_size = attn_size
        self.n_heads = n_heads

        xavier_scale = np.sqrt((emb_size + attn_size*n_heads)/2)
        w_q = torch.randn(emb_size, attn_size*n_heads)/xavier_scale
        self.w_q = nn.Parameter(w_q)
        w_k = torch.randn(emb_size, attn_size*n_heads)/xavier_scale
        self.w_k = nn.Parameter(w_k)
        w_v = torch.randn(emb_size, attn_size*n_heads)/xavier_scale
        self.w_v = nn.Parameter(w_v)

        modules = []
        modules.append(nn.Linear(seq_len*attn_size*n_heads,
                                 seq_len*attn_size*n_heads))
        if bnorm:
            modules.append(nn.BatchNorm1d(seq_len*attn_size*n_heads))
        modules.append(globals()[act_fxn]())
        modules.append(nn.Linear(seq_len*attn_size*n_heads, emb_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(emb_size))
        modules.append(globals()[act_fxn]())
        modules.append(nn.Linear(emb_size, emb_size))
        self.enc_net = nn.Sequential(*modules)

    def forward(self, q, k, v):
        """
        q: float tensor (B,S,E)
            the queries
        k: float tensor (B,S,E)
            the keys
        v: float tensor (B,S,E)
            the values
        """
        fq = torch.matmul(q,self.w_q) # (B,S,A)
        fk = torch.matmul(k,self.w_k) # (B,S,A)
        fv = torch.matmul(v,self.w_v) # (B,S,A)

        batch,seq,_ = q.shape
        atn = self.attn_size
        fq = fq.reshape(batch, seq, self.n_heads, self.attn_size) 
        fq = fq.permute(0,2,1,3) # (B,H,S,A)
        fk = fk.reshape(batch, seq, self.n_heads, self.attn_size) 
        fk = fk.permute(0,2,3,1) # (B,H,A,S)
        fv = fv.reshape(batch, seq, self.n_heads, self.attn_size) 
        fv = fv.permute(0,2,1,3) # (B,H,S,A)

        scale = 1/np.sqrt(atn)
        attn = scale*torch.matmul(fq, fk) # (B,H,S,S)
        ps = F.softmax(attn, dim=-1)
        fv = torch.matmul(ps,fv) # (B,H,S,A)
        fv = fv.permute(0,2,1,3).reshape(batch,seq*atn*self.n_heads)
        embs = self.enc_net(fv)
        return embs

class TransAutoencoder(CustomBase):
    def __init__(self, *args, **kwargs):
        """
        See CustomBase for arguments
        """
        super().__init__(*args, **kwargs)
        self.transformer_type = AUTOENCODER

        self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        self.encoder = crabmods.Encoder(self.enc_slen,
                                             emb_size=self.emb_size,
                                             attn_size=self.attn_size,
                                             n_layers=self.enc_layers,
                                             n_heads=self.n_heads,
                                             use_mask=self.enc_mask,
                                             act_fxn=self.act_fxn)
        if self.collapse_type == TRANS:
            self.collapser = CollapsingAttention(emb_size=self.emb_size,
                                            seq_len=self.enc_slen,
                                            attn_size=self.attn_size,
                                            n_heads=self.n_heads,
                                            bnorm=self.class_bnorm,
                                            act_fxn=self.act_fxn)
        else:
            self.collapser = ReductionConvolution(emb_size=self.emb_size,
                                                  n_filts=self.n_filts)

        if self.expand_type == TRANS:
            self.expander = ExpandingAttention(emb_size=self.emb_size,
                                            seq_len=self.dec_slen,
                                            attn_size=self.attn_size,
                                            n_heads=self.n_heads,
                                            bnorm=self.class_bnorm,
                                            act_fxn=self.act_fxn)
        else:
            self.expander = ExpandingRSSM(seq_len=self.dec_slen,
                                          emb_size=self.emb_size,
                                          h_size=self.emb_size,
                                          attn_size=self.attn_size,
                                          n_heads=self.n_heads)

        self.decoder = crabmods.Decoder(self.dec_slen, self.emb_size,
                                             self.attn_size,
                                             self.dec_layers,
                                             n_heads=self.n_heads,
                                             use_mask=False,
                                             act_fxn=self.act_fxn)

        self.classifier = crabmods.Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)

        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.collapse_dropout = nn.Dropout(self.collapse_drop_p)
        self.expand_dropout = nn.Dropout(self.expand_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def forward(self, x, y):
        self.embeddings.weight.data[0,:] = 0 # Mask index
        embs = self.embeddings(x)
        encs = self.encoder(embs)
        encs = self.enc_dropout(encs)
        encs = self.collapser(encs,encs,encs)
        encs = self.collapse_dropout(encs)
        encs = self.expander(encs)
        encs = self.expand_dropout(encs)
        dembs = self.embeddings(y)
        decs = self.decoder(dembs, encs)
        decs = self.dec_dropout(decs)
        decs = decs.reshape(-1,decs.shape[-1])
        return self.classifier(decs)

class Codt(CustomBase):
    """
    this is an experimental model to prove that the basic idea of
    minimal supervision transformers can learn compression. Collapsing
    is performed using a decoding module with a learned initialization
    vector
    """
    def __init__(self, collapse_size=1, collapse_layers=3,
                                        *args, **kwargs):
        """
        collapse_size: int
            the number of elements in the collapsed vector sequence
        collapse_layers: int
            the number of layers used to collapse the encoded sequence
        """
        super().__init__(*args, **kwargs)
        if kwargs['dataset'] == "WebstersDictionary":
            self.transformer_type = DICTIONARY
        else:
            self.transformer_type = AUTOENCODER
        self.collapse_size = collapse_size
        self.collapse_layers = collapse_layers
        self.dec_slen = self.enc_slen

        self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        self.encoder = crabmods.Encoder(self.enc_slen,
                                             emb_size=self.emb_size,
                                             attn_size=self.attn_size,
                                             n_layers=self.enc_layers,
                                             n_heads=self.n_heads,
                                             use_mask=self.enc_mask,
                                             act_fxn=self.act_fxn)
        self.collapse_init = torch.randn(1,1,self.emb_size)
        self.collapse_init = nn.Parameter(self.collapse_init)
        self.collapser = crabmods.Decoder(self.collapse_size,
                                 emb_size=self.emb_size,
                                 attn_size=self.attn_size,
                                 n_layers=3,
                                 n_heads=self.n_heads,
                                 use_mask=False,
                                 act_fxn=self.act_fxn)

        use_mask = not self.init_decs and self.ordered_preds
        self.decoder = crabmods.Decoder(self.dec_slen,self.emb_size,
                                             self.attn_size,
                                             self.dec_layers,
                                             n_heads=self.n_heads,
                                             use_mask=use_mask,
                                             init_decs=self.init_decs,
                                             act_fxn=self.act_fxn)

        self.classifier = crabmods.Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)

        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.collapse_dropout = nn.Dropout(self.collapse_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def forward(self, x, y, **kwargs):
        self.embeddings.weight.data[0,:] = 0 # Mask index
        x_mask = (x==0).masked_fill(x==0,1e-10)
        embs = self.embeddings(x)
        encs = self.encoder(embs,x_mask=x_mask)
        encs = self.enc_dropout(encs)
        init = self.collapse_init.repeat((len(x),self.collapse_size,1))
        encs = self.collapser(init, encs)
        encs = self.collapse_dropout(encs)
        y_mask = (y==0).float().masked_fill(y==0,1e-10)
        embs = self.embeddings(y)
        decs = self.decoder(embs, encs, x_mask=y_mask)
        decs = self.dec_dropout(decs)
        shape = decs.shape
        decs = decs.reshape(-1,decs.shape[-1])
        preds = self.classifier(decs)
        if self.transformer_type == DICTIONARY:
            return preds.reshape(shape[0],shape[1],-1),encs
        return preds.reshape(shape[0],shape[1],-1)

class LSTMBaseline(CustomBase):
    """
    this is a simple lstm autoencoder
    """
    def __init__(self, *args, **kwargs):
        """
        collapse_size: int
            the number of elements in the collapsed vector sequence
        collapse_layers: int
            the number of layers used to collapse the encoded sequence
        """
        super().__init__(*args, **kwargs)
        self.transformer_type = AUTOENCODER
        self.dec_slen = self.enc_slen

        self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        self.encoder = nn.LSTMCell(input_size=self.emb_size,
                                   hidden_size=self.emb_size)
        self.decoder = nn.LSTMCell(input_size=self.emb_size,
                                   hidden_size=self.emb_size)

        self.classifier = crabmods.Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)

        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.collapse_dropout = nn.Dropout(self.collapse_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def get_blank_h(self,bsize):
        """
        bsize: int
        """
        h = torch.zeros(bsize, self.emb_size)
        if next(self.parameters).is_cuda:
            h = h.cuda()
        c = torch.zeros_like(h)
        return (h,c)

    def forward(self, x, **kwargs):
        self.embeddings.weight.data[0,:] = 0 # Mask index
        embs = self.embeddings(x)
        h = self.get_blank_h(len(x))
        for i in range(x.shape[1]):
            h = self.encoder(embs[:,i],h)
        preds = []
        for i in range(x.shape[1]):
            pred = self.classifier(h[0])
            preds.append(pred[:,None])
            emb = self.embeddings(torch.argmax(preds,dim=-1))
            h = self.decoder(emb,h)
        return torch.cat(preds,dim=1)

