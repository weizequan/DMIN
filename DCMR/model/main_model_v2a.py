import torch
from torch import nn
import torch.nn.functional as F
from .models_v2a import New_Audio_Guided_Attention
from .models_v2a import EncoderLayer, Encoder, DecoderLayer, Decoder
from torch.nn import MultiheadAttention


class InternalTemporalRelationModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(InternalTemporalRelationModule, self).__init__()
        self.encoder_layer = EncoderLayer(d_model=d_model, nhead=4)
        self.encoder = Encoder(self.encoder_layer, num_layers=2)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)
        # add relu here?

    def forward(self, feature):
        # feature: [seq_len, batch, dim]
        feature = self.affine_matrix(feature)
        feature = self.encoder(feature)

        return feature


class CrossModalRelationAttModule(nn.Module):
    def __init__(self, input_dim, d_model):
        super(CrossModalRelationAttModule, self).__init__()

        self.decoder_layer = DecoderLayer(d_model=d_model, nhead=4)
        self.decoder = Decoder(self.decoder_layer, num_layers=1)

        self.affine_matrix = nn.Linear(input_dim, d_model)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, query_feature, memory_feature):
        query_feature = self.affine_matrix(query_feature)
        output = self.decoder(query_feature, memory_feature)

        return output

class SupvLocalizeModule(nn.Module):
    def __init__(self, d_model):
        super(SupvLocalizeModule, self).__init__()
        # self.affine_concat = nn.Linear(2*256, 256)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(d_model, 1) # start and end
        self.event_classifier = nn.Linear(d_model, 28)
        self.event_self = nn.Linear(d_model, 28)
       # self.softmax = nn.Softmax(dim=-1)

    def forward(self, fused_content,self_visual):

        max_fused_content, _ = fused_content.transpose(1, 0).max(1)
        max_self_content, _ = self_visual.transpose(1, 0).max(1)
        logits = self.classifier(fused_content)
        # scores = self.softmax(logits)
        class_logits = self.event_classifier(max_fused_content)
        class_self = self.event_self(max_self_content)
        class_scores = class_logits

        return logits, class_scores, class_self


class AudioVideoInter(nn.Module):
    def __init__(self, d_model, n_head, head_dropout=0.1):
        super(AudioVideoInter, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.video_multihead = MultiheadAttention(d_model, num_heads=n_head, dropout=head_dropout)
        self.norm1 = nn.LayerNorm(d_model)


    def forward(self, video_feat, audio_feat,labels):
        t_size,batch,channel=video_feat.shape
        video_mean = []
        #cross matching mechanism
        for i in range(batch):
            video_sum=[]
            for j in range(t_size):        
                if(labels[i][j]==1):            
                    temp = torch.zeros([channel]).cuda()
                    temp = temp.double()
                    temp =video_feat[j,i,:]
                    video_sum.append(temp.unsqueeze(0))
            video_sum = torch.cat(video_sum,0)
            video_mean = video_sum.mean(dim=-1)
            t = video_mean.size(0)
            video_mean = video_mean.unsqueeze(-1)
            k=0
            while k+t<=t_size:
                audio_feat[k:k+t,i,:]*=video_mean
                k=k+1
        return  audio_feat                     
                    
class supv_main_model(nn.Module):
    def __init__(self):
        super(supv_main_model, self).__init__()

        self.spatial_channel_att = New_Audio_Guided_Attention().cuda()
        self.video_input_dim = 512 
        self.video_fc_dim = 512
        self.d_model = 256
        self.v_fc = nn.Linear(self.video_input_dim, self.video_fc_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

        self.video_encoder = InternalTemporalRelationModule(input_dim=512, d_model=256)
        self.video_decoder = CrossModalRelationAttModule(input_dim=512, d_model=256)
        self.audio_encoder = InternalTemporalRelationModule(input_dim=128, d_model=256)
        self.audio_decoder = CrossModalRelationAttModule(input_dim=128, d_model=256)

        self.AVInter = AudioVideoInter(self.d_model, n_head=4, head_dropout=0.1)
        self.localize_module = SupvLocalizeModule(self.d_model)


    def forward(self, visual_feature, audio_feature,labels):
        # [batch, 10, 512]

        # optional, we add a FC here to make the model adaptive to different visual features (e.g., VGG ,ResNet)
        audio_feature = audio_feature.transpose(1, 0).contiguous()

        # audio-guided needed
        visual_feature = visual_feature.transpose(1, 0).contiguous()

        # audio query
        video_key_value_feature = self.video_encoder(visual_feature)
        audio_query_output = self.audio_decoder(audio_feature, video_key_value_feature)

        # video query
        audio_key_value_feature = self.audio_encoder(audio_feature)
        video_query_output = self.video_decoder(visual_feature, audio_key_value_feature)

        video_query_output= self.AVInter(video_query_output, audio_query_output,labels)
        scores = self.localize_module(video_query_output,video_key_value_feature)
        return scores
