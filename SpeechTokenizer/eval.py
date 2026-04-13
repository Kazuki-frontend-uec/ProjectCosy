def evaluate_codebook_usage(model_path, codebook_size, filelist_txt, device="cuda"):
    """
    保存済みモデルをロードし、コードブック利用率とperplexityを計算
    """
    # モデル読み込み
    tokenizer = SpeechTokenizer(
        input_dim=512, hidden_dim=512, codebook_size=codebook_size
    ).to(device)
    tokenizer.load_state_dict(torch.load(model_path, map_location=device))
    tokenizer.eval()

    # データセット準備
    ds = AudioDataset(filelist_txt)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    whisper_enc = WhisperEncoder("base", device=device).to(device)
    whisper_enc.eval()

    all_indices = []
    for wav in tqdm(dl, desc="Evaluating"):
        wav = wav.to(device)
        with torch.no_grad():
            emb = whisper_enc(wav)
            emb = emb.unsqueeze(0)
            _, _, _, idx = tokenizer(emb)
            all_indices.append(idx.flatten())

    usage_rate, perplexity = compute_codebook_stats(all_indices, codebook_size)
    print(f"Usage Rate: {usage_rate*100:.2f}%  |  Perplexity: {perplexity:.2f}")
    return usage_rate, perplexity
