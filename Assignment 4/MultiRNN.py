import torch
import math

torch.set_default_dtype(torch.double)

class MultiRNN:

    """
        Consists of only input and the computed hidden states are the outputs, which will be inputs for the next such layer.
        The last RNN should have only one output, that is only the final hidden state, after which we would apply a Linear Layer.
    """

    def __init__(self, num_in, num_hidden, num_out, activation=torch.tanh):
        
        """
            num_in = size of the one-hot encoded input "word". One element of such a batch will have many such "words".
        """
        
        self.Wxh = torch.randn(num_hidden, num_in)
        self.Wxh = self.Wxh * math.pow(2 / (num_in + num_hidden), 0.5)
        self.Bxh = torch.zeros(num_hidden, 1)

        self.Whh = torch.randn(num_hidden, num_hidden)
        self.Whh = self.Whh * math.pow(2 / (num_hidden + num_hidden), 0.5)
        self.Bhh = torch.zeros(num_hidden, 1)

        self.activation = activation

        self.gradWxh = torch.zeros_like(self.Wxh)
        self.gradBxh = torch.zeros_like(self.Bxh)

        self.gradWhh = torch.zeros_like(self.Whh)
        self.gradBhh = torch.zeros_like(self.Bhh)

    def cuda(self):
        
        """
            For transferring to GPU device
        """
        
        self.Wxh = self.Wxh.cuda()
        self.Bxh = self.Bxh.cuda()

        self.Whh = self.Whh.cuda()
        self.Bhh = self.Bhh.cuda()

        
        self.gradWxh = self.gradWxh.cuda()
        self.gradBxh = self.gradBxh.cuda()

        self.gradWhh = self.gradWhh.cuda()
        self.gradBhh = self.gradBhh.cuda()

        return self

    def forward(self, input):

        """
            Assuming input is (batch_size, seq_len, num_input) and 
            output is required to be (batch_size, seq_len, num_out)
            assuming within a batch we have fixed length sequences

            We are assuming that the hidden state of every layer 
            is initialised to fixed zero tensors
        """

        batch_size   = input.shape[0]
        seq_length   = input.shape[1]
        hid_length   = self.Bhh.shape[0]

        self.hidden_state = torch.zeros(batch_size, seq_length, hid_length).to(self.Wxh.device)

        for seq in range(seq_length):

            bat_seq_inp  = input[:, seq, :]
            prev_hidden  = self.hidden_state[:, max(0, seq-1), :]
            self.hid_inp = torch.matmul(bat_seq_inp, torch.t(self.Wxh)) + torch.t(self.Bxh)
            self.hid_hid = torch.matmul(prev_hidden, torch.t(self.Whh)) + torch.t(self.Bhh)
            self.hid_tot = self.hid_inp + self.hid_hid

            # Can use ReLU instead ?
            self.hidden_state[:, seq, :] = self.activation(self.hid_tot)

        output = self.hidden_state + 0
        return output

    def backward(self, input, gradOutput):

        """
            input is (batch_size, seq_len, num_in)
            gradOutput is (batch_size, seq_len, num_out) as per previous assignment, 
            we divide by batch_size here and then sum when it comes to individual layers
        """

        ## Clearing the gradients from the last iteration ##
        
        self.gradWxh = torch.zeros_like(self.Wxh)
        self.gradBxh = torch.zeros_like(self.Bxh)

        self.gradWhh = torch.zeros_like(self.Whh)
        self.gradBhh = torch.zeros_like(self.Bhh)

        seq_length  = self.hidden_state.shape[1]
        gradInput   = torch.zeros_like(input)
        gradOut     = torch.zeros_like(gradOutput[:, seq_length-1, :]) 

        for seq in range(seq_length-1,-1,-1):
            
            gradOut += gradOutput[:, seq, :]
            ## First differentiating wrt the activation function, assuming tanh here
            inp = self.hidden_state[:, seq, :]
            grad_tanh = 1 - inp**2
            gradOut = grad_tanh * gradOut

            prev_hidden = torch.zeros(batch_size, hidden_state)
            if seq > 0:
                prev_hidden = self.hidden_state[:, seq-1, :]
            
            self.gradWhh += torch.t(torch.matmul(torch.t(prev_hidden), gradOut))
            self.gradBhh += torch.t(torch.sum(gradOut, dim=0).unsqueeze(0))

            inp = input[:, seq, :]

            self.gradWxh += torch.t(torch.matmul(torch.t(inp), gradOut))
            self.gradBxh += torch.t(torch.sum(gradOut, dim=0).unsqueeze(0))


            gradInput[:, seq, :] = torch.matmul(gradOut, self.Wxh)
            gradOut              = torch.matmul(gradOut, self.Whh)

        return gradInput
