def _attention_value(self, q, k, v, mask, b, src_s, tgt_s, n_head, head_dim):
        attn_weights = self._attention_weights(q, k, mask, b, src_s, n_head)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================
        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]
        # v.shape :[ 257, b, n_heads_per_partition, head_dim] (257, 4, 6, 257)
        # context layer shape: [b, np, sq, hn]
        output_size = (v.size(1),v.size(2),q.size(0),v.size(3)) # (4,6,1,64)
        v = v.reshape(v.size(0),v.size(1)*v.size(2),v.size(3)) # (257, 24, 64)
        # change view [b * np, sq, sk]
        attention_probs = attn_weights.view(output_size[0] * output_size[1],output_size[2], -1)
         # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, v.transpose(0, 1)) # (24,1,64)
        # change view [b, np, sq, hn] (4,6,1,64)
        context_layer = context_layer.view(*output_size)
        # [b, np, sq, hn] --> [sq, b, np, hn] =(1, 4,6,64)
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]   = (1,4,384)
        new_context_layer_shape = context_layer.size()[:-2] + \
            (context_layer.size()[-2]*context_layer.size()[-1],)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer # (1,4,384)
        