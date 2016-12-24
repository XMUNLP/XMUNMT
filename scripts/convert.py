# convert.py
# convert GroundHog's format to our format
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import numpy
import cPickle
import argparse


def loadvocab(name):
    fd = open(name, "r")
    vocab = cPickle.load(fd)
    fd.close()
    return vocab


def invertvoc(vocab):
    v = {}
    for k, idx in vocab.iteritems():
        v[idx] = k

    return v


def getoption():
    option = {}

    option["corpus"] = None
    option["vocab"] = None

    # model parameters
    option["embdim"] = [620, 620]
    option["hidden"] = [1000, 1000, 1000]
    option["maxpart"] = 2
    option["maxhid"] = 500
    option["deephid"] = 620

    # tuning options
    option["alpha"] = 5e-4
    option["batch"] = 128
    option["momentum"] = 0.0
    option["optimizer"] = "rmsprop"
    option["variant"] = "graves"
    option["norm"] = 1.0
    option["stop"] = 0
    option["decay"] = 0.5

    # runtime information
    option["cost"] = 0
    option["count"] = 0
    option["epoch"] = 0
    option["maxepoch"] = 5
    option["sort"] = 20
    option["shuffle"] = False
    option["limit"] = [50, 50]
    option["freq"] = 1000
    option["vfreq"] = 1000
    option["sfreq"] = 50
    option["seed"] = 1234
    option["validation"] = None
    option["references"] = None
    option["bleu"] = 0.0
    option["indices"] = None

    # beam search
    option["beamsize"] = 10
    option["normalize"] = False
    option["maxlen"] = None
    option["minlen"] = None

    # special symbols
    option["unk"] = "UNK"
    option["eos"] = "<eos>"

    return option


def get_rnnsearch_keys():
    keys = []

    keys.append("rnnsearch/source_embedding/embedding")
    keys.append("rnnsearch/source_embedding/bias")
    keys.append("rnnsearch/target_embedding/embedding")
    keys.append("rnnsearch/target_embedding/bias")
    keys.append("rnnsearch/encoder/forward/gru_cell/reset_gate/matrix_0")
    keys.append("rnnsearch/encoder/forward/gru_cell/reset_gate/matrix_1")
    keys.append("rnnsearch/encoder/forward/gru_cell/update_gate/matrix_0")
    keys.append("rnnsearch/encoder/forward/gru_cell/update_gate/matrix_1")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/encoder/forward/gru_cell/candidate/bias")
    keys.append("rnnsearch/encoder/backward/gru_cell/reset_gate/matrix_0")
    keys.append("rnnsearch/encoder/backward/gru_cell/reset_gate/matrix_1")
    keys.append("rnnsearch/encoder/backward/gru_cell/update_gate/matrix_0")
    keys.append("rnnsearch/encoder/backward/gru_cell/update_gate/matrix_1")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/encoder/backward/gru_cell/candidate/bias")
    keys.append("rnnsearch/decoder/initial/matrix_0")
    keys.append("rnnsearch/decoder/initial/bias")
    keys.append("rnnsearch/decoder/attention/attention_w/matrix_0")
    keys.append("rnnsearch/decoder/attention/query_w/matrix_0")
    keys.append("rnnsearch/decoder/attention/attention_v/matrix_0")
    keys.append("rnnsearch/decoder/gru_cell/reset_gate/matrix_0")
    keys.append("rnnsearch/decoder/gru_cell/reset_gate/matrix_1")
    keys.append("rnnsearch/decoder/gru_cell/reset_gate/matrix_2")
    keys.append("rnnsearch/decoder/gru_cell/update_gate/matrix_0")
    keys.append("rnnsearch/decoder/gru_cell/update_gate/matrix_1")
    keys.append("rnnsearch/decoder/gru_cell/update_gate/matrix_2")
    keys.append("rnnsearch/decoder/gru_cell/candidate/matrix_0")
    keys.append("rnnsearch/decoder/gru_cell/candidate/matrix_1")
    keys.append("rnnsearch/decoder/gru_cell/candidate/matrix_2")
    keys.append("rnnsearch/decoder/gru_cell/candidate/bias")
    keys.append("rnnsearch/decoder/maxout/matrix_0")
    keys.append("rnnsearch/decoder/maxout/matrix_1")
    keys.append("rnnsearch/decoder/maxout/matrix_2")
    keys.append("rnnsearch/decoder/maxout/bias")
    keys.append("rnnsearch/decoder/deepout/matrix_0")
    keys.append("rnnsearch/decoder/logits/matrix_0")
    keys.append("rnnsearch/decoder/logits/bias")


def get_groundhog_keys():
    keys = []

    keys.append("W_0_enc_approx_embdr")
    keys.append("b_0_enc_approx_embdr")
    keys.append("W_0_dec_approx_embdr")
    keys.append("b_0_dec_approx_embdr")
    keys.append("W_0_enc_reset_embdr_0")
    keys.append("R_enc_transition_0")
    keys.append("W_0_enc_update_embdr_0")
    keys.append("G_enc_transition_0")
    keys.append("W_0_enc_input_embdr_0")
    keys.append("W_enc_transition_0")
    keys.append("b_0_enc_input_embdr_0")
    keys.append("W_0_back_enc_reset_embdr_0")
    keys.append("R_back_enc_transition_0")
    keys.append("W_0_back_enc_update_embdr_0")
    keys.append("G_back_enc_transition_0")
    keys.append("W_0_back_enc_input_embdr_0")
    keys.append("W_back_enc_transition_0")
    keys.append("b_0_back_enc_input_embdr_0")
    keys.append("W_0_dec_initializer_0")
    keys.append("b_0_dec_initializer_0")
    keys.append("A_dec_transition_0")
    keys.append("B_dec_transition_0")
    keys.append("D_dec_transition_0")
    keys.append("W_0_dec_reset_embdr_0")
    keys.append("W_0_dec_dec_reseter_0")
    keys.append("R_dec_transition_0")
    keys.append("W_0_dec_update_embdr_0")
    keys.append("W_0_dec_dec_updater_0")
    keys.append("G_dec_transition_0")
    keys.append("W_0_dec_input_embdr_0")
    keys.append("W_0_dec_dec_inputter_0")
    keys.append("W_dec_transition_0")
    keys.append("b_0_dec_input_embdr_0")
    keys.append("W_0_dec_hid_readout_0")
    keys.append("W_0_dec_prev_readout_0")
    keys.append("W_0_dec_repr_readout")
    keys.append("b_0_dec_hid_readout_0")
    keys.append("W1_dec_deep_softmax")
    keys.append("W2_dec_deep_softmax")
    keys.append("b_dec_deep_softmax")

    return keys


def parseargs():
    msg = "convert groudhog's model to our format"
    parser = argparse.ArgumentParser(description=msg)

    msg = "search state"
    parser.add_argument("--state", required=True, help=msg)
    msg = "search model"
    parser.add_argument("--model", required=True, help=msg)
    msg = "output"
    parser.add_argument("--output", required=True, help=msg)

    return parser.parse_args()


def main(args):
    fd = open(args.state)
    state = cPickle.load(fd)
    fd.close()
    option = getoption()

    if not state["search"]:
        raise ValueError("only support RNNsearch architecture")

    embdim = state["rank_n_approx"]
    hidden_dim = state["dim"]

    option["embdim"] = [embdim, embdim]
    option["hidden"] = [hidden_dim, hidden_dim, hidden_dim]
    option["maxpart"] = int(state["maxout_part"])
    option["maxhid"] = hidden_dim / 2
    option["deephid"] = embdim
    option["vocab"] = [state["word_indx"], state["word_indx_trgt"]]

    option["source_eos_id"] = state["null_sym_source"]
    option["target_eos_id"] = state["null_sym_target"]

    svocab = loadvocab(option["vocab"][0])
    tvocab = loadvocab(option["vocab"][1])
    isvocab = invertvoc(svocab)
    itvocab = invertvoc(tvocab)

    svocab[option["eos"]] = state["null_sym_source"]
    tvocab[option["eos"]] = state["null_sym_target"]
    isvocab[state["null_sym_source"]] = option["eos"]
    itvocab[state["null_sym_target"]] = option["eos"]

    option["bosid"] = 0
    option["eosid"] = state["null_sym_target"]

    if len(isvocab) != state["n_sym_source"]:
        raise ValueError("source vocab size not match")

    if len(itvocab) != state["n_sym_target"]:
        raise ValueError("target vocab size not match")

    option["vocabulary"] = [[svocab, isvocab], [tvocab, itvocab]]

    option["unk"] = state["oov"]
    option["batch"] = state["bs"]
    option["limit"] = [state["seqlen"], state["seqlen"]]
    option["sort"] = state["sort_k_batches"]
    option["seed"] = state["seed"]
    option["shuffle"] = state["shuffle"]

    params = numpy.load(args.model)
    params = dict(params)

    if len(params) != 40:
        raise ValueError("configuration not supported")

    rkeys = get_rnnsearch_keys()
    gkeys = get_groundhog_keys()

    pval = {}

    for key1, key2 in zip(rkeys, gkeys):
        pval[key1] = params[key2]

    fd = open(args.output, "w")
    cPickle.dump(option, fd)
    cPickle.dump(rkeys, fd)
    numpy.savez(fd, **pval)
    fd.close()


if __name__ == "__main__":
    args = parseargs()
    main(args)
