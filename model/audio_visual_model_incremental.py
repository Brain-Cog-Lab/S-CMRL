import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import LSCLinear, SplitLSCLinear
from .prompt import EPrompt
from .attention import Prompt_Attention, BilinearPooling, CrossModalAttention, InternalTemporalRelationModule, CrossModalRelationAttModule


class IncreAudioVisualNet(nn.Module):
    def __init__(self, args, step_out_class_num, LSC=False):
        super(IncreAudioVisualNet, self).__init__()
        self.args = args
        self.modality = args.modality
        self.num_classes = step_out_class_num
        self.use_e_prompt = args.e_prompt
        if self.modality != 'visual' and self.modality != 'audio' and self.modality != 'audio-visual':
            raise ValueError('modality must be \'visual\', \'audio\' or \'audio-visual\'')
        if self.modality == 'visual':
            self.visual_proj = nn.Linear(768, 768)
        elif self.modality == 'audio':
            self.audio_proj = nn.Linear(768, 768)
        else:
            self.audio_proj = nn.Linear(768, 768)
            self.visual_proj = nn.Linear(768, 768)
            self.attn_audio_proj = nn.Linear(768, 768)
            self.attn_visual_proj = nn.Linear(768, 768)
        
        if LSC:
            self.classifier = LSCLinear(768, self.num_classes)
        else:
            self.classifier = nn.Linear(768, self.num_classes)

        if self.use_e_prompt is True:
            task_num = args.num_classes // args.class_num_per_step
            self.a_prompt = EPrompt(key_dim=768, prompt_dim=args.prompt_dim, pool_size=task_num)
            self.v_prompt = EPrompt(key_dim=768, prompt_dim=args.prompt_dim, pool_size=task_num)
            self.a_prompt_attention = Prompt_Attention(dim=768, prompt_dim=args.prompt_dim)
            self.v_prompt_attention = Prompt_Attention(dim=768, prompt_dim=args.prompt_dim)
            self.av_cue_fusion = CrossModalAttention(768, 768, 768)

            self.visual_encoder = InternalTemporalRelationModule(input_dim=768, d_model=768, feedforward_dim=1024)  # self.video_fc_dim
            self.visual_decoder = CrossModalRelationAttModule(input_dim=768, d_model=768, feedforward_dim=512)
            self.audio_encoder = InternalTemporalRelationModule(input_dim=768, d_model=768, feedforward_dim=1024)  # self.video_fc_dim
            self.audio_decoder = CrossModalRelationAttModule(input_dim=768, d_model=768, feedforward_dim=512)
    
    def forward(self, visual=None, audio=None, out_logits=True, out_features=False, out_features_norm=False, out_feature_before_fusion=False, out_attn_score=False, AFC_train_out=False, is_train=True, task_id=0, task_ids_for_kl=None):
        if self.modality == 'visual':
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            visual_feature = torch.mean(visual, dim=1)
            visual_feature = F.relu(self.visual_proj(visual_feature))
            logits = self.classifier(visual_feature)
            outputs = ()
            if AFC_train_out:
                visual_feature.retain_grad()
                outputs += (logits, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(visual_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

        elif self.modality == 'audio':
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')
            audio_feature = F.relu(self.audio_proj(audio))
            logits = self.classifier(audio_feature)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                outputs += (logits, audio_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if out_features:
                    outputs += (F.normalize(audio_feature),)
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs
        else:
            if visual is None:
                raise ValueError('input frames are None when modality contains visual')
            if audio is None:
                raise ValueError('input audio are None when modality contains audio')

            visual = visual.view(visual.shape[0], 8, -1, 768)  # [b, l, s, d] -> [256, 8, 196, 768]
            spatial_attn_score, temporal_attn_score = self.audio_visual_attention(audio, visual)
            visual_pooled_feature = torch.sum(spatial_attn_score * visual, dim=2)
            visual_pooled_feature = torch.sum(temporal_attn_score * visual_pooled_feature, dim=1)

            # 这里是新加的对e_prompt的操作
            if self.use_e_prompt:
                if is_train:
                    prompt_mask = task_id  # 当前的重新计算, 之前的用之前的

                    visual_feature = torch.sum(F.softmax(visual, dim=2) * visual, dim=2)  # ikd
                    visual_feature = torch.sum(F.softmax(visual_feature, dim=1) * visual_feature, dim=1)
                    audio_feature = audio
                    v_res = self.v_prompt(visual_feature, prompt_mask=prompt_mask, cls_features=visual_feature)  # 输入和查询的是一样的
                    a_res = self.a_prompt(audio_feature, prompt_mask=prompt_mask, cls_features=audio_feature)  # 输入和查询的是一样的

                    visual_feature = visual_feature.unsqueeze(1).transpose(1, 0).contiguous()  # (seq, batch, dim)
                    audio_feature = audio.unsqueeze(1).transpose(1, 0).contiguous()  # (seq, batch, dim)

                    # audio query
                    visual_key_value_feature = self.visual_encoder(visual_feature)
                    audio_query_output = self.audio_decoder(audio_feature, visual_key_value_feature).squeeze()  # (batch, dim)

                    # video query
                    audio_key_value_feature = self.audio_encoder(audio_feature)
                    visual_query_output = self.visual_decoder(visual_feature, audio_key_value_feature).squeeze()  # (batch, dim)

                    visual_feature = visual_feature.squeeze()  # (batch, dim)
                    audio_feature = audio_feature.squeeze()  # (batch, dim)

                    self.v_prompt.prompt[0] = audio_query_output.max(dim=0)[0]

                    cross_visual_feature = visual_feature + self.v_prompt.prompt[0]

                    self.a_prompt.prompt[0] = visual_query_output.max(dim=0)[0]

                    cross_audio_feature = audio_feature + self.a_prompt.prompt[0]
                    # cross_audio_feature = audio_feature

                    audio_visual_features = self.av_cue_fusion(cross_audio_feature, cross_visual_feature)

                else:
                    prompt_mask = None


                if task_ids_for_kl is not None:
                    prompt_mask = task_ids_for_kl

                    v_res_for_kl = self.v_prompt(visual_feature, prompt_mask=prompt_mask, cls_features=visual_feature)  # 输入和查询的是一样的
                    v_prompt_for_kl = v_res_for_kl['batched_prompt']

                    cross_visual_feature_for_kl = self.v_prompt_attention(audio_feature, visual_feature, v_prompt_for_kl)
                    # cross_visual_feature_for_kl = visual_feature

                    a_res_for_kl = self.a_prompt(audio_feature, prompt_mask=prompt_mask, cls_features=audio_feature)  # 输入和查询的是一样的
                    a_prompt_for_kl = a_res_for_kl['batched_prompt']

                    cross_audio_feature_for_kl = self.a_prompt_attention(visual_feature, audio_feature, a_prompt_for_kl)
                    # cross_audio_feature_for_kl = audio_feature

                    audio_visual_features_for_kl = self.av_cue_fusion(cross_audio_feature_for_kl, cross_visual_feature_for_kl)
                    # cross_visual_feature_for_kl = F.relu(self.visual_proj(cross_visual_feature_for_kl))
                    # cross_audio_feature_for_kl = F.relu(self.audio_proj(cross_audio_feature_for_kl))
                    # audio_visual_features_for_kl = cross_visual_feature_for_kl + cross_audio_feature_for_kl

                # audio_visual_features = cross_visual_feature + cross_audio_feature
            else:
                audio_feature = F.relu(self.audio_proj(audio))
                visual_feature = F.relu(self.visual_proj(visual_pooled_feature))

                audio_visual_features = visual_feature + audio_feature
            
            logits = self.classifier(audio_visual_features)
            outputs = ()
            if AFC_train_out:
                audio_feature.retain_grad()
                visual_feature.retain_grad()
                visual_pooled_feature.retain_grad()
                outputs += (logits, visual_pooled_feature, audio_feature, visual_feature)
                return outputs
            else:
                if out_logits:
                    outputs += (logits,)
                if task_ids_for_kl is not None:
                    outputs += (self.classifier(audio_visual_features_for_kl), )
                if out_features:
                    if out_features_norm:
                        outputs += (F.normalize(audio_visual_features),)
                    else:
                        outputs += (audio_visual_features,)
                if out_feature_before_fusion:
                    outputs += (F.normalize(audio_feature), F.normalize(visual_feature))
                if out_attn_score:
                    outputs += (spatial_attn_score, temporal_attn_score)
                if is_train and self.use_e_prompt:
                    outputs += (a_res['reduce_sim'], v_res['reduce_sim'])
                if len(outputs) == 1:
                    return outputs[0]
                else:
                    return outputs

    def audio_visual_attention(self, audio_features, visual_features):

        proj_audio_features = torch.tanh(self.attn_audio_proj(audio_features))
        proj_visual_features = torch.tanh(self.attn_visual_proj(visual_features))

        # (BS, 8, 14*14, 768)
        spatial_score = torch.einsum("ijkd,id->ijkd", [proj_visual_features, proj_audio_features])
        # (BS, 8, 14*14, 768)
        spatial_attn_score = F.softmax(spatial_score, dim=2)
        # (BS, 8, 768)
        spatial_attned_proj_visual_features = torch.sum(spatial_attn_score * proj_visual_features, dim=2)

        # (BS, 8, 768)
        temporal_score = torch.einsum("ijd,id->ijd", [spatial_attned_proj_visual_features, proj_audio_features])
        temporal_attn_score = F.softmax(temporal_score, dim=1)

        return spatial_attn_score, temporal_attn_score
    

    def incremental_classifier(self, numclass):
        weight = self.classifier.weight.data
        bias = self.classifier.bias.data
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features

        self.classifier = nn.Linear(in_features, numclass, bias=True)
        self.classifier.weight.data[:out_features] = weight
        self.classifier.bias.data[:out_features] = bias
