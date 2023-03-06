from config import *


def create_embedding(multi_res, inputs):
    embedding_input = lambda x: x
    period_fn = [torch.sin, torch.cos]
    embeddings = []
    out_dim = 0
    max_freq_log2 = multi_res - 1
    if nerf_embedder['include_input']:
        embeddings.append(embedding_input)
        out_dim += nerf_embedder['input_dim']
    if nerf_embedder['log_sampling']:
        freq_bands = 2. ** torch.linspace(0., max_freq_log2, steps=multi_res)
    else:
        freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, steps=multi_res)
    for freq in freq_bands:
        for period in period_fn:
            embedding_freq = lambda x, p_fn=period, freq=freq: p_fn(x * freq)
            embeddings.append(embedding_freq)
            out_dim += nerf_embedder['input_dim']
    return torch.cat([embedding(inputs) for embedding in embeddings], -1), out_dim

def get_embeddings(inputs, viewdirs): # inputs are (sample_points, viewdirs) after render_rays
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embeddings, _ = create_embedding(nerf_embedder['multi_res'], inputs_flat)
    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs, __ = create_embedding(nerf_embedder['multi_res_views'], input_dirs_flat)
        embeddings = torch.cat([embeddings, embedded_dirs], -1)
    return embeddings