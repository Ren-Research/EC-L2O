import torch
import torch.nn as nn

## ML model
class LSTM_unroll(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, 
                 seq_length, w, l_1, l_2, l_3):
        
        super(LSTM_unroll, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        self.w = w
        self.l_1 = l_1
        self.l_2 = l_2
        self.l_3 = l_3

    def forward(self, x, mode="train", calib = True):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size, device = x.device)
        
        action_0 = torch.zeros(x.size(0), 1, x.size(2), device = x.device)
        action_sequence = []
        
        seq_length = x.size(1)
        
        
        if (mode not in ["train", "val"]):
            raise NotImplementedError
        
        if (seq_length != self.seq_length and mode == "train"):
            print("The length of sequence is changed, be careful!!!!")
            assert 0
            
        for i in range(seq_length):
            x_i = x[:,i,:].unsqueeze(1)
            input_i = torch.cat([x_i, action_0], 2)
            ula, (h_out, c_out) = self.lstm(input_i, (h_0, c_0))
            
            ula = ula.view(-1, self.hidden_size)
            out = self.fc(ula).unsqueeze(1)
            
            h_0 = h_out
            c_0 = c_out
            
            if i < 1:
                # The initial action directly follow the demand
                action_0 = x_i
            else:
                action_prev = action_sequence[-1]
                if calib:
                    action_0 = (1 + self.w*self.l_2)*x_i + self.w*self.l_1*action_prev + self.w*self.l_3*out
                    action_0 /= 1 + self.w*(self.l_1 + self.l_2 + self.l_3)
                else:
                    action_0 = out
            
            action_sequence.append(action_0)

        final_out = torch.cat(action_sequence,1)
        
        return final_out
