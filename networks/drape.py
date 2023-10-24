import torch.nn as nn
import torch
import numpy as np

class skip_connection(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        width,
        depth,
        weight_norm=True,
        skip_layer=[],
    ):
        super().__init__()

        dims = [d_in] + [width] * depth + [d_out]
        self.num_layers = len(dims)

        self.skip_layer = skip_layer

        for l in range(0, self.num_layers - 1):

            if l in self.skip_layer:
                lin = torch.nn.Linear(dims[l] + dims[0], dims[l+1])
            else:
                lin = torch.nn.Linear(dims[l], dims[l+1])

            if weight_norm:
                lin = torch.nn.utils.weight_norm(lin)
            else:
                torch.nn.init.xavier_uniform_(lin.weight)
                torch.nn.init.zeros_(lin.bias)


            setattr(self, "lin" + str(l), lin)

        self.activation = torch.nn.LeakyReLU()


    def forward(self, input, softmax=False):
        """MPL query.

        Tensor shape abbreviation:
            B: batch size
            D: input dimension
            
        Args:
            input (tensor): network input. shape: [B, D]

        Returns:
            output (tensor): network output. Might contains placehold if mask!=None shape: [B, ?]
        """

        batch_size, n_dim = input.shape

        x = input
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_layer:
                x = torch.cat([x, input], -1)
            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)

        return x


class DisDecoder_linear(nn.Module):
    def __init__(
        self,
        d_lin=200*200*6,
        d_in=85,
        d_hidden=256,
        depth=6,
        skip_layer=[3]
    ):
        super().__init__()
        
        self.d_lin = d_lin
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.depth = depth
        self.skip_layer = skip_layer

        self.skip_connection = skip_connection(d_in=d_in, d_out=d_hidden, width=d_hidden, depth=depth, skip_layer=skip_layer)
        self.linear = nn.Linear(d_hidden, d_lin)

        torch.nn.init.xavier_uniform_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input, infer=False):

        batch_size = input.shape[0]
        
        x = self.skip_connection(input)
        
        x = self.linear(x)
        x = x.reshape(batch_size, -1, 200, 200)

        return x


class Pred_decoder_uv_linear(nn.Module):
    def __init__(self, d_in=82, d_hidden=512, depth=8, skip_layer=[], tanh=False):
        super(Pred_decoder_uv_linear, self).__init__()

        self.decoder = DisDecoder_linear(d_in=d_in, d_hidden=d_hidden, depth=depth, skip_layer=skip_layer)

    def forward(self, pose):
        uv_disp = self.decoder(pose)
        return uv_disp[:,:3], uv_disp[:,3:]
