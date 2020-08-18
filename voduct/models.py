import torch
import torch.nn as nn
from torch.nn import ReLU
import numpy as np
import time
import os
import torch.nn.functional as F
import sencoder as sen
import matplotlib.pyplot as plt
import crab.custom_modules as crabmods

DEVICE_DICT = {-1:"cpu", 0:"cuda:0"}

SEQ2SEQ = "seq2seq"
AUTOENCODER = "autoencoder"
TRANSFORMER_TYPE = {"Transformer":SEQ2SEQ,
                    "TransAutoencoder":AUTOENCODER,
                    "Codt":AUTOENCODER}
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

class TransformerBase(nn.Module):
    def __init__(self, seq_len=None, n_vocab=None, emb_size=512,
                                         enc_slen=None,
                                         dec_slen=None,
                                         attn_size=64,
                                         n_heads=8,
                                         enc_layers=6,
                                         dec_layers=6,
                                         enc_mask=False,
                                         class_h_size=4000,
                                         class_bnorm=True,
                                         class_drop_p=0,
                                         act_fxn="ReLU",
                                         collapse_type=CONV,
                                         expand_type=TRANS,
                                         enc_drop_p=0,
                                         dec_drop_p=0,
                                         collapse_drop_p=0,
                                         expand_drop_p=0,
                                         n_filts=10,
                                         ordered_preds=False,
                                         gen_decs=False,
                                         **kwargs):
        """
        seq_len: int or None
            the length of the sequences to be analyzed. If None,
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
        """
        super().__init__()

        self.seq_len = seq_len
        self.enc_slen = enc_slen if enc_slen is not None else seq_len
        self.dec_slen = dec_slen if dec_slen is not None else seq_len
        self.n_vocab = n_vocab
        self.emb_size = emb_size
        self.attn_size = attn_size
        self.n_heads = n_heads
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.enc_mask = enc_mask
        self.class_bnorm = class_bnorm
        self.class_drop_p = class_drop_p
        self.class_h_size = class_h_size
        self.act_fxn = act_fxn
        self.collapse_type = collapse_type
        self.expand_type = expand_type
        self.enc_drop_p = enc_drop_p
        self.collapse_drop_p = collapse_drop_p
        self.expand_drop_p = expand_drop_p
        self.dec_drop_p = dec_drop_p
        self.n_filts = n_filts
        self.ordered_preds = ordered_preds
        self.gen_decs = gen_decs

class Transformer(TransformerBase):
    def __init__(self, *args, **kwargs):
        """
        See TransformerBase for arguments
        """
        super().__init__(*args, **kwargs)
        self.transformer_type = SEQ2SEQ

        self.embeddings = nn.Embedding(self.n_vocab, self.emb_size)

        self.encoder = crabmods.Encoder(self.enc_slen,
                                            emb_size=self.emb_size,
                                            attn_size=self.attn_size,
                                            n_layers=self.enc_layers,
                                            n_heads=self.n_heads,
                                            use_mask=self.enc_mask,
                                            act_fxn=self.act_fxn)
        self.decoder = crabmods.Decoder(self.dec_slen,self.emb_size,
                                            self.attn_size,
                                            self.dec_layers,
                                            n_heads=self.n_heads,
                                            act_fxn=self.act_fxn,
                                            use_mask=self.ordered_preds,
                                            gen_decs=self.gen_decs)

        self.classifier = Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)
        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def forward(self, x, y):
        """
        x: float tensor (B,S)
        y: float tensor (B,S)
        """
        self.embeddings.weight.data[0,:] = 0 # Mask index
        embs = self.embeddings(x)
        encs = self.encoder(embs)
        encs = self.enc_dropout(encs)
        dembs = self.embeddings(y)
        decs = self.decoder(dembs, encs)
        decs = self.dec_dropout(decs)
        decs = decs.reshape(-1,decs.shape[-1])
        preds = self.classifier(decs)
        return preds.reshape(len(y),y.shape[1],preds.shape[-1])

class Classifier(nn.Module):
    def __init__(self, emb_size, n_vocab, h_size, bnorm=True,
                                                  drop_p=0,
                                                  act_fxn="ReLU"):
        """
        emb_size: int
            the size of the embedding layer
        n_vocab: int
            the number of words in the vocabulary
        h_size: int
            the size of the hidden layer
        bnorm: bool
            if true, the hidden layers use a batchnorm layer
        drop_p: float
            the dropout probability of the dropout modules
        act_fxn: str
            the name of the activation function to be used with the
            MLP
        """

        super().__init__()
        self.emb_size = emb_size
        self.n_vocab = n_vocab
        self.h_size = h_size
        self.bnorm = bnorm
        self.drop_p = drop_p
        self.act_fxn = act_fxn

        modules = []
        modules.append(nn.Linear(emb_size,h_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.Dropout(drop_p))
        modules.append(globals()[act_fxn]())

        modules.append(nn.Linear(h_size,h_size))
        if bnorm:
            modules.append(nn.BatchNorm1d(h_size))
        modules.append(nn.Dropout(drop_p))
        modules.append(globals()[act_fxn]())

        modules.append(nn.Linear(h_size,n_vocab))
        self.classifier = nn.Sequential(*modules)

    def forward(self, x):
        return self.classifier(x)

class Sin(nn.Module):
    def __init__(self):
        """
        A sinusoidal activation function.
        """
        super().__init__()

    def forward(self, x):
        return torch.sin(x)




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

class TransAutoencoder(TransformerBase):
    def __init__(self, *args, **kwargs):
        """
        See TransformerBase for arguments
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
                                             act_fxn=self.act_fxn)

        self.classifier = Classifier(self.emb_size,
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

class Codt(TransformerBase):
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

        self.decoder = crabmods.Decoder(self.dec_slen,self.emb_size,
                                             self.attn_size,
                                             self.dec_layers,
                                             n_heads=self.n_heads,
                                             act_fxn=self.act_fxn)
        self.decode_init = torch.randn(1,1,self.emb_size)
        self.decode_init = nn.Parameter(self.decode_init)

        self.classifier = Classifier(self.emb_size,
                                     self.n_vocab,
                                     h_size=self.class_h_size,
                                     bnorm=self.class_bnorm,
                                     drop_p=self.class_drop_p,
                                     act_fxn=self.act_fxn)

        self.enc_dropout = nn.Dropout(self.enc_drop_p)
        self.collapse_dropout = nn.Dropout(self.collapse_drop_p)
        self.dec_dropout = nn.Dropout(self.dec_drop_p)

    def forward(self, x, y):
        self.embeddings.weight.data[0,:] = 0 # Mask index
        embs = self.embeddings(x)
        encs = self.encoder(embs)
        encs = self.enc_dropout(encs)
        init = self.collapse_init.repeat((len(x),self.collapse_size,1))
        encs = self.collapser(init, encs)
        encs = self.collapse_dropout(encs)
        init = self.decode_init.repeat((len(x),self.dec_slen,1))
        decs = self.decoder(init, encs)
        decs = self.dec_dropout(decs)
        shape = decs.shape
        decs = decs.reshape(-1,decs.shape[-1])
        preds = self.classifier(decs)
        return preds.reshape(shape[0],shape[1],-1)

