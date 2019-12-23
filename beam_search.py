import torch
import torch.nn as nn
from utils import subsequent_mask


class BeamSearch(nn.Module):
    def __init__(self, model, beam_size, max_seq_len,
                 src_pad_idx, trg_pad_idx, trg_bos_idx, trg_eos_idx):
        super(BeamSearch, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.src_pad_idx = src_pad_idx
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx

        self.model = model
        self.model.eval()

        self.register_buffer("init_seq", torch.LongTensor([[trg_bos_idx]]).cuda())
        self.register_buffer(
            "blank_seqs",
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long).cuda())
        self.register_buffer(
            "len_map",
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0).cuda())

    def _model_decode(self, comment, memory, code_mask):
        comment_mask = subsequent_mask(comment.size(1)).cuda()
        dec_output, _, _ = self.model.decode(memory, code_mask, comment, comment_mask)
        return self.model.generator(dec_output)

    def _get_init_state(self, seq, parent_matrix, brother_matrix,
                        relative_parent_ids, relative_brother_ids,
                        code_mask):
        beam_size = self.beam_size

        enc_output, _ = self.model.encode(seq, parent_matrix, brother_matrix,
                                           relative_parent_ids, relative_brother_ids)

        dec_output = self._model_decode(self.init_seq, enc_output, code_mask)

        best_k_probs, best_k_ids = dec_output[:, -1].topk(beam_size)

        scores = best_k_probs.unsqueeze(0)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 0] = self.trg_bos_idx
        gen_seq[:, 1] = best_k_ids[0]
        # enc_output = enc_output.repeat(beam_size, 1, 1)
        return enc_output, gen_seq, scores

    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        beam_size = self.beam_size

        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[:, -1, :].topk(beam_size)
        best_k2_probs = best_k2_probs.view(beam_size, -1) + scores.transpose(0, 1)

        # Get the best k candidates from k^2 candidates.
        best_k_probs, best_k_idx_in_k2 = best_k2_probs.view(-1).topk(beam_size)
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]

        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        gen_seq[:, step] = best_k_idx

        scores = best_k_probs.unsqueeze(0)

        return gen_seq, scores

    def beam_search(self, seq, parent_matrix, brother_matrix, relative_parent_ids, relative_brother_ids):

        src_pad_idx, trg_eos_idx = self.src_pad_idx, self.trg_eos_idx
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha

        with torch.no_grad():
            code_mask = (seq != src_pad_idx).unsqueeze(-2)
            enc_output, gen_seq, scores = self._get_init_state(seq, parent_matrix, brother_matrix,
                                                               relative_parent_ids, relative_brother_ids, code_mask)

            ans_idx = 0  # default
            for step in range(2, max_seq_len):  # decode up to max length
                gen_seq = gen_seq.view(beam_size, -1, self.max_seq_len)
                out = []
                for i in range(beam_size):
                    dec_out = self._model_decode(gen_seq[i, :, :step], enc_output, code_mask)
                    out.append(dec_out)
                # dec_output = self._model_decode(gen_seq[:, :step], enc_output, code_mask)
                dec_output = torch.cat(out, dim=0)
                gen_seq = gen_seq.view(-1, self.max_seq_len)
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(1 - eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(1)
                    ans_idx = ans_idx.item()
                    break
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
